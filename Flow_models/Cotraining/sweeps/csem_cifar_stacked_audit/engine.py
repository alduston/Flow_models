"""Guarded adaptation of pinned CIFAR training; keep inner routing and optimizer intact."""
import ast, inspect
import core
from stacked import vae_factory
from band_regularizer import adapt_band_source,band_namespace


def adapt(source):
    counts=dict(factory=0,encode=0,base=0,resample=0,multiplier=0,resume=0)
    class A(ast.NodeTransformer):
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='VAE':
                counts['factory']+=1;n.func.id='make_vae'
            return n
        def visit_Assign(self,n):
            if len(n.targets)==1:
                target=ast.unparse(n.targets[0]);val=ast.unparse(n.value)
                if target=='(mu_base, logvar_base)' and val=='vae.encode(x)':counts['base']+=1;return None
                if target=='(mu, logvar)' and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='vae.encode':
                    counts['encode']+=1;return ast.parse('mu, logvar, z0 = vae.encode_sample(x)').body[0]
                if target=='z0' and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='vae.reparameterize':
                    counts['resample']+=1;return None
                if target=='encoder_score_is_detached':
                    counts['multiplier']+=1
                    return [ast.parse("encoder_csem_gradient_multiplier *= cfg['rep_multiplier']").body[0],n]
                if target=='ema_decay':
                    return [n,*ast.parse("cfg['model_parameter_counts']={'vae':sum(p.numel() for p in vae.parameters()),'score':sum(p.numel() for p in unet_lsi.parameters())}").body]
            return self.generic_visit(n)
        def visit_Expr(self,n):
            if ast.unparse(n) == "print('--> Starting Dual Co-training...')":
                return [*ast.parse('if start_epoch == 0: run_initial_audit(0,vae,unet_lsi_ema,train_l.dataset,test_l.dataset,cfg,device)').body,n]
            if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='validate_resume_config':
                counts['resume']+=1;return [n,ast.parse("validate_stack_resume(state['cfg'],cfg)").body[0]]
            return self.generic_visit(n)
    tree=ast.fix_missing_locations(A().visit(ast.parse(source)))
    assert counts==dict(factory=1,encode=2,base=1,resample=2,multiplier=1,resume=1),counts
    return ast.unparse(tree),counts

def validate_stack_resume(old,new):
    for k in ('bundle_recipe_hash','blocks','topology','rep_multiplier','fixed_band_weight'):
        if old.get(k)!=new.get(k):raise ValueError('Resume mismatch: '+k)

def fixed_band(logvar,floor):
    from band_regularizer import tmin_band_penalty
    return tmin_band_penalty(logvar,floor)

def train(cfg):
    from evaluation import evaluate
    from audit import run_audit
    source,counts=adapt(inspect.getsource(core.train_vae_cotrained_cond))
    source,bc=adapt_band_source(source)
    # B is an independent, fixed control when encoder pressure is varied.
    old='band_objective = active_score_w_vae * band_penalty'
    assert source.count(old)==1
    source=source.replace(old,"band_objective = cfg['fixed_band_weight'] * band_penalty")
    ns=dict(core.__dict__,**band_namespace());ns.update(make_vae=lambda **kw:vae_factory(cfg,**kw),
        evaluate_current_state=evaluate,validate_stack_resume=validate_stack_resume,run_initial_audit=run_audit)
    exec(compile(source,'<cifar-joint-stack>','exec'),ns)
    cfg.update(stack_adapter_counts=counts,band_adapter_counts=bc)
    return ns['train_vae_cotrained_cond'](cfg)
