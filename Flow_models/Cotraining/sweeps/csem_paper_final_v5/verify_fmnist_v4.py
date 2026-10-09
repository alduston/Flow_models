#!/usr/bin/env python3
"""CPU checks for configuration/metric/target routing; no Torch dependency."""
import ast, csv, importlib.util, math, sys, types, textwrap
from pathlib import Path
import numpy as np
import pandas as pd

base=Path(__file__).resolve().parent
core_source=(base/'csem_paper_core_v1.py').read_text()
tree=ast.parse(core_source)
# Load only preset data and resolver functions; keep the full GPU preflight separate.
core=types.ModuleType('csem_paper_core_v1')
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'DATASET_PRESETS','DATASET_TO_PRESET'} for t in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),'<presets>','exec'),core.__dict__)
for n in tree.body:
    if isinstance(n,ast.FunctionDef) and n.name=='resolve_model_preset':
        exec(compile(ast.Module(body=[n],type_ignores=[]),'<resolver>','exec'),core.__dict__)
core.resolve_bespoke_fid_classifier=lambda dataset,override: False
sys.modules['csem_paper_core_v1']=core
sys.path.insert(0,str(base))
from validate_csem_paper_final_v1 import main as validate
validate()

spec=importlib.util.spec_from_file_location('compiler',base/'compile_csem_paper_final_v1.py')
compiler=importlib.util.module_from_spec(spec);spec.loader.exec_module(compiler)
# Combined CSVs have all-NaN columns belonging to another dataset/CFG/time.
r=pd.Series({'fid_rk4_25_randtok_cfg1_5_temp1_initgaussianT1p5':np.nan,
 'fid_rk4_25_randtok_cfg3_0_temp1_initgaussianTK2':2.,
 'fid_rk4_25_randtok_cfg3_0_initoracleqT2':1.,
 'fid_rk4_25_randtok_cfg3_0_temp1_initgaussianT2':5.24,
 'fid_rk4_20_randtok_cfg3_0_temp1_initgaussianT2':6.})
assert compiler._find_metric(r,'fid_rk4_',25)[1]==5.24
assert math.isnan(compiler._find_metric(r,'fid_rk4_',40)[1])
records=pd.DataFrame([{'tag':'LSI_Diff','stage':'cotrain','epoch':700},
 {'tag':'Ctrl_Diff','stage':'cotrain','epoch':700}])
assert {x['regime'].iloc[0] for x in compiler._main_regime(records,'independent_pair')}=={'Independent CSEM','Independent Tweedie'}

class Tensor(np.ndarray):
    def detach(self): return self
    def flatten(self,start_dim=0): return self.reshape(self.shape[0],-1) if start_dim==1 else self.reshape(-1)
def tensor(x): return np.asarray(x,dtype=float).view(Tensor)
seen=[]
def control(x,t,y):
    seen.append((x.copy(),t.copy()))
    return tensor(np.zeros_like(x))
# Execute the actual tracking preparation/loss code. At T_K=0 the representation
# is z0,t=0,noise=0; the detached head must see z_t_head,t_head,noise_head instead.
start=core_source.index('                    # --- Tracking head (trains on detached latents) ---',core_source.index('def train_vae_cotrained_cond'))
end=core_source.index('                    opt_tracking.zero_grad()',start)
tracking=textwrap.dedent(core_source[start:end])
scope={'z_t':tensor([[0.],[0.]]),'z_mu_t':tensor([[0.],[0.]]),'t':tensor([0.,0.]),
 'eps_target_control':tensor([[0.],[0.]]),'eps_target_lsi':tensor([[0.],[0.]]),
 'z_t_head':tensor([[1.],[2.]]),'t_head':tensor([.5,1.]),
 'alpha_head':tensor([[.8],[.6]]),'mu_head':tensor([[0.],[0.]]),
 'sigma_head':tensor([[.6],[.8]]),'noise_head':tensor([[1.],[2.]]),
 'eps_target_control_head':tensor([[1.],[2.]]),'eps_target_lsi_head':tensor([[.2],[.3]]),
 'score_head_time_weights':tensor([1.,1.]),'cotrain_head':'lsi','cos_w':0.,
 'cfg':{'score_w':1.},'y_in':tensor([1.,2.]),'unet_control':control,
 'weighted_score_prediction_losses':lambda pred,target,w: (float(np.mean((pred-target)**2)),0.)}
exec(compile(tracking,'<actual_tracking_block>','exec'),scope)
assert np.allclose(seen[0][0],scope['z_t_head'])
assert np.allclose(seen[0][1],scope['t_head'])
assert math.isclose(scope['tracking_loss'],2.5)
assert '_randtok_cfg' in (base/'csem_paper_suite_v1.py').read_text()
print('PASS: 22 resolved configs; Gaussian/T_K/oracle/NaN/solver filtering; both independent regimes; actual full-horizon Tweedie tracking target.')
print('Torch/CUDA training and FID reproduction require the cluster GPU environment.')
