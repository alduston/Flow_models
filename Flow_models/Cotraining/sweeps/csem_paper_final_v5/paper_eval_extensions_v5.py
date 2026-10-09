"""Paper-only evaluation additions; original August-13 training source stays intact."""
import ast
import copy
import inspect
import random
from contextlib import contextmanager
import numpy as np

@contextmanager
def fixed_metric_rng(torch, seed):
    py_state=random.getstate(); np_state=np.random.get_state()
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed)); random.seed(int(seed)); np.random.seed(int(seed))
        try: yield
        finally: random.setstate(py_state); np.random.set_state(np_state)

def feature_sw2(real, fake, cfg, compute):
    import torch
    # Feature-space distance is distinct from the legacy latent-space SW2.
    n=int(cfg.get('feature_sw2_n_projections',256))
    g=torch.Generator(device='cpu').manual_seed(int(cfg.get('seed',42))+33333)
    theta=torch.randn(real.shape[1],n,generator=g).to(real.device,real.dtype)
    theta=theta/theta.norm(dim=0,keepdim=True).clamp_min(1e-12)
    return max(float(compute(real,fake,n_projections=n,theta=theta)),0.0)**0.5

_gap_cache={}
def score_unit_gap(original, net, mus, logvars, cfg, device):
    import torch
    if original not in _gap_cache:
        tree=ast.parse(inspect.getsource(original))
        fn=tree.body[0];fn.name='_paper_score_unit_gap'
        count=0
        for n in ast.walk(fn):
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='score_gap_per_sample' for t in n.targets):
                n.value=ast.parse('(eps_diff_sq / sigma_sq).sum(dim=(1,2,3))',mode='eval').body;count+=1
        if count!=2: raise RuntimeError('Unexpected CSEM diagnostic source; refuse silent epsilon-gap substitution')
        ns=dict(original.__globals__);exec(compile(ast.fix_missing_locations(tree),'<paper_score_gap>','exec'),ns)
        _gap_cache[original]=ns[fn.name]
    with fixed_metric_rng(torch,int(cfg.get('seed',42))+44444):
        return _gap_cache[original](net,mus,logvars,cfg,device,labels=None,
            num_classes=cfg.get('num_classes'),num_samples=int(cfg.get('eval_lsi_gap_samples',2500)),
            num_time_points=int(cfg.get('eval_lsi_gap_time_points',50)),batch_size=int(cfg['batch_size']))

def install_historical_evaluator(module, source_path):
    original=module.evaluate_current_state
    tree=ast.parse(source_path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_current_state')
    fn=copy.deepcopy(fn);fn.name='_paper_extra_evaluate'
    # Replace only the evaluator's sampler list and its unet extension block.
    for j,n in enumerate(fn.body):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='configs' for t in n.targets):
            fn.body[j]=ast.parse("configs = [{'method':'VAE_Rec_eps','steps':0,'desc':'Recon (posterior z)','use_rand_token':False}] + cfg['_paper_extra_samplers']").body[0]
            if not (isinstance(fn.body[j+1],ast.If) and ast.unparse(fn.body[j+1].test)=='unet is not None'):
                raise RuntimeError('Unexpected historical evaluation structure')
            del fn.body[j+1];break
    code=ast.unparse(ast.fix_missing_locations(fn))
    code=code.replace('fid = compute_fid_from_features(real_features, fake_features)',
        'feature_w2 = feature_sw2(real_features, fake_features, cfg, compute_sw2)\n        fid = compute_fid_from_features(real_features, fake_features)')
    code=code.replace("'w2': w2,", "'w2': w2, 'feature_w2': feature_w2,")
    for typ in ['rk4','heun']:
        line=f"output_dict[f'sw2_{typ}_{{col_suffix}}'] = r['w2']"
        code=code.replace(line,line+f"\n            output_dict[f'feature_sw2_{typ}_{{col_suffix}}'] = r['feature_w2']")
    code=code.replace("output_dict['sw2_vae_recon'] = r['w2']", "output_dict['sw2_vae_recon'] = r['w2']\n            output_dict['feature_sw2_vae_recon'] = r['feature_w2']")
    code=code.replace("output_dict['lsi_gap_unet_uncond'] = lsi_gap_unet", "output_dict['lsi_gap_unet_uncond'] = lsi_gap_unet\n    output_dict['csem_gap_score_uncond'] = score_unit_gap(compute_lsi_gap, unet, encoder_mus, encoder_logvars, cfg, device)")
    ns=module.__dict__;ns.update(feature_sw2=feature_sw2,score_unit_gap=score_unit_gap)
    exec(compile(code,'<aug13_paper_eval_only>','exec'),ns)
    extra=ns['_paper_extra_evaluate'];signature=inspect.signature(original)
    def evaluate(*args,**kwargs):
        result=original(*args,**kwargs)
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();kw=dict(bound.arguments)
        cfg=copy.deepcopy(kw['cfg'])
        cfg['_paper_extra_samplers']=[dict(method=method,steps=steps,desc='Paper comparison',use_rand_token=True,cfg_level=g)
            for g in dict.fromkeys(float(x) for x in cfg['paper_cfg_grid'])
            for method,steps in [('rk4_ode',20),('rk4_ode',25),('heun_sde',20)]]
        kw['cfg']=cfg
        # Extra metrics cannot consume the training RNG or alter original metrics.
        with fixed_metric_rng(module.torch,int(cfg.get('seed',42))+55555): additions=extra(**kw)
        result.update({k:v for k,v in additions.items() if k not in result})
        return result
    module.evaluate_current_state=evaluate
