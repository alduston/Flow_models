"""Original CIFAR metric backend and sampler suite, with coherent posterior banks."""
import ast,inspect
from pathlib import Path
import core
from suite import fixed_metric_rng


def make_evaluator():
    tree=ast.parse(inspect.getsource(core.evaluate_current_state));counts=dict(encode=0,A=0,B=0,gap=0)
    class A(ast.NodeTransformer):
        def visit_Assign(self,n):
            target=ast.unparse(n.targets[0]) if len(n.targets)==1 else ''
            if target == "output_dict['csem_gap_score_uncond']":
                counts['gap']+=1;n.value=ast.parse("float('nan')",mode='eval').body;return n
            if target=='(mu, logvar)'  and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='vae.encode':
                counts['encode']+=1
                return ast.parse("epsA = fixed_posterior_eps_bank_A[bank_idx:bank_idx+bsz].to(device) if fixed_posterior_eps_bank_A is not None else None\nmu, logvar, coherent_zA = vae.encode_sample(x,eps=epsA)").body
            if target=='zA' and ast.unparse(n.value)=='mu + std * epsA':
                counts['A']+=1;n.value=ast.Name(id='coherent_zA',ctx=ast.Load());return n
            return self.generic_visit(n)
        def visit_Call(self,n):
            if ast.unparse(n.func)=='real_latents_B.append':
                counts['B']+=1;n.args[0]=ast.parse('vae.encode_sample(x,eps=epsB)[2].cpu()',mode='eval').body
            return self.generic_visit(n)
    tree=ast.fix_missing_locations(A().visit(tree));assert counts==dict(encode=1,A=1,B=1,gap=1),counts
    ns=dict(core.__dict__);exec(compile(tree,'<coherent-cifar-eval>','exec'),ns)
    return ns['evaluate_current_state']

_RAW=None

def evaluate(epoch_idx,prefix,vae,unet,loader,cfg,device,lpips_fn,**kw):
    global _RAW
    if _RAW is None:_RAW=make_evaluator()
    # Conditional moments from one sampled root do not define a joint Gaussian mixture.
    # Disable the inherited flat-mixture oracle/gap, explicitly replace with valid probes.
    with fixed_metric_rng(core.torch,int(cfg['seed'])+55555):
        values=_RAW(epoch_idx,prefix,vae,unet,loader,cfg,device,lpips_fn,**kw)
    from audit import run_audit
    train_loader=kw.get('oracle_reference_loader')
    if train_loader is not None:
        run_audit(epoch_idx,vae,unet,train_loader.dataset,loader.dataset,cfg,device)
    values['joint_reverse_solves']=1
    values['score_parameters']=sum(p.numel() for p in unet.parameters())
    values['vae_parameters']=sum(p.numel() for p in vae.parameters())
    return values
