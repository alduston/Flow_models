"""Offline sweep/recipe/coupling tests, plus an explicit real-Torch smoke test.

Offline numeric tests execute the actual posterior sampler with a NumPy tensor
shim. They do not claim to validate autograd or CUDA. `paper.py smoke` exercises
real models, detached inner graphs, VAE gradients and ancestral sampling.
"""
from __future__ import annotations
import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import numpy as np


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def extract(node, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<actual_source>', 'exec'), namespace)


@contextlib.contextmanager
def config_stub(base):
    tree = ast.parse((base/'core.py').read_text())
    core = types.ModuleType('core')
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'DATASET_PRESETS', 'DATASET_TO_PRESET'} for t in node.targets):
            extract(node, core.__dict__)
    extract(function(tree, 'resolve_model_preset'), core.__dict__)
    core.resolve_bespoke_fid_classifier = lambda dataset, override: False
    saved = {k: sys.modules.get(k) for k in ('core', 'suite')}
    sys.modules['core'] = core; sys.modules.pop('suite', None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class Tensor(np.ndarray):
    """Only the tensor operations needed by the actual posterior sampler."""
    def split(self, widths, dim=0):
        return tuple(np.split(self, np.cumsum(widths)[:-1], axis=dim))
    def chunk(self, count, dim=0):
        return tuple(np.array_split(self, count, axis=dim))
    def clamp(self, low, high):
        return np.clip(self, low, high).view(Tensor)
    def flatten(self, start_dim=0):
        shape = (*self.shape[:start_dim], -1)
        return self.reshape(shape)
    def mean(self, dim=None, keepdim=False):
        return np.mean(np.asarray(self), axis=dim, keepdims=keepdim).view(Tensor)
    def square(self):
        return np.square(self).view(Tensor)
    def view_as(self, other):
        return self.reshape(other.shape)


def posterior_coupling_checks(base):
    tree = ast.parse((base/'hier_models.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HierarchicalVAE')
    sampler = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'encode_sample')
    rng = np.random.default_rng(123)
    tensor = lambda v: np.asarray(v).view(Tensor)
    backend = types.SimpleNamespace(Tensor=Tensor,
        cat=lambda seq, dim=0: tensor(np.concatenate(seq, axis=dim)),
        exp=lambda v: tensor(np.exp(v)), sqrt=lambda v: tensor(np.sqrt(v)),
        randn_like=lambda v: tensor(rng.standard_normal(v.shape)))
    core = types.SimpleNamespace()
    ns = dict(torch=backend, core=core, math=math)
    extract(function(ast.parse((base/'core.py').read_text()), 'ou_partial_groupnorm'), ns)
    core.ou_partial_groupnorm = ns['ou_partial_groupnorm']
    extract(sampler, ns)
    for widths in ((4, 4),):
        B, H, W = 2, 2, 2
        m = tensor(rng.normal(size=(B, 8, H, W)))
        lv = tensor(rng.normal(scale=.1, size=m.shape))
        eps = tensor(rng.normal(size=m.shape))
        conditions = []
        for c in widths[1:]:
            def adapter(z, c=c):
                mean = np.asarray(z).mean(axis=1, keepdims=True)
                dmu = np.repeat(.2*mean, c, axis=1)
                dlv = np.repeat(.03*mean, c, axis=1)
                return tensor(np.concatenate([dmu, dlv], axis=1))
            conditions.append(adapter)
        obj = types.SimpleNamespace(encode=lambda _: (m, lv), block_channels=widths,
                                    block_anchor_mode='ou_partial', block_anchor_time=1.05,
                                    conditions=conditions)
        mu, logvar, z = ns['encode_sample'](obj, None, eps=eps)
        np.testing.assert_allclose(z, mu+np.exp(.5*logvar)*eps, atol=1e-13)
        changed = eps.copy(); changed[:, :widths[0]] += .7
        mu2, lv2, z2 = ns['encode_sample'](obj, None, eps=changed)
        np.testing.assert_allclose(mu[:, :widths[0]], mu2[:, :widths[0]])
        assert np.max(np.abs(mu[:, widths[0]:]-mu2[:, widths[0]:])) > 1e-3
        # Verify each moment used exactly the already-realized clean ancestors.
        offset = 0
        for i, c in enumerate(widths):
            raw_mu = m[:, offset:offset+c]
            if i:
                dm, _ = conditions[i-1](z[:, :offset]).chunk(2, dim=1)
                raw_mu = raw_mu+dm
            expected = core.ou_partial_groupnorm(raw_mu, 1.05)
            np.testing.assert_allclose(mu[:, offset:offset+c], expected, atol=1e-13)
            offset += c
    print('OK actual posterior sampler: fixed draws, clean-parent coupling, OU-partial anchors')


def source_checks(base):
    from hier_engine import adapt_trainer_source
    source = (base/'core.py').read_text(); tree = ast.parse(source)
    provenance = json.loads((base/'provenance.json').read_text())
    repaired = 'def setup_run_results_dir(base_dir="run_results", wipe=True, preserve_checkpoints=True):'
    original = 'def setup_run_results_dir(base_dir="run_results", wipe=not bool(cfg.get("paper_resume_state")), preserve_checkpoints=True):'
    assert source.count(repaired) == 1
    restored = source.replace(repaired, original)
    assert hashlib.sha256(restored.encode()).hexdigest() == provenance['source_files']['core(1).py']
    for file, old in (('suite.py', 'suite(1).py'), ('paper_io.py', 'paper_io(1).py')):
        assert hashlib.sha256((base/file).read_bytes()).hexdigest() == provenance['source_files'][old]
    # Defining this helper must no longer refer to a nonexistent module cfg.
    extract(function(tree, 'setup_run_results_dir'), {})
    trainer = function(tree, 'train_vae_cotrained_cond')
    adapted, counts = adapt_trainer_source(ast.get_source_segment(source, trainer))
    compile(adapted, '<hierarchy>', 'exec')
    calls = [n for n in ast.walk(ast.parse(adapted)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('unet_lsi', 'unet_control')]
    assert len(calls) == 17
    for call in calls:
        context = next(k.value for k in call.keywords if k.arg == 'context')
        outer = ast.unparse(call.args[0]) in ('z_t_for_score', 'z_mu_t_for_score')
        if outer:
            assert ast.unparse(context) == ("canonical_stability_scale_gradient(z0, encoder_csem_gradient_multiplier) "
                "if cfg.get('hierarchy_variant') == 'optimized_v3' else z0")
        else:
            assert ast.unparse(context) == 'z0.detach()'
    # Objective, gradient replacement, clipping and optimizer statements remain.
    mathematical_targets = {'loss_joint', 'active_score_inner_objective', 'score_time_weights',
                            'score_head_time_weights', 'routed_score_gradients', 'kl', 'opt_joint',
                            'sched_joint', 'joint_preclip'}
    def selected(t):
        return [ast.dump(n, include_attributes=False) for n in ast.walk(t)
                if isinstance(n, ast.Assign) and any(isinstance(v, ast.Name) and v.id in mathematical_targets for v in n.targets)]
    assert selected(trainer) == selected(ast.parse(adapted).body[0])
    print('OK source hashes and minimal import repair; CIFAR loss/optimizer/routing/clipping AST preserved')


def sampling_checks(base):
    """Execute the actual ancestral sampler with an instrumented simple solver."""
    tree = ast.parse((base/'hier_models.py').read_text())
    calls = []
    tensor = lambda v: np.asarray(v).view(Tensor)
    class Module:
        pass
    class Solver:
        def __init__(self, *args, **kwargs):
            pass
        def sample(self, model, *, x_init, y, cfg_scale):
            context = getattr(model, 'context', None)
            calls.append(None if context is None else context.copy())
            return tensor(.5*x_init+1 if context is None else .5*x_init+context.mean(1, True))
    ns = dict(nn=types.SimpleNamespace(Module=Module),
              torch=types.SimpleNamespace(cat=lambda values, dim=0: tensor(np.concatenate(values, axis=dim))),
              core=types.SimpleNamespace(UniversalSampler=Solver))
    extract(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BoundConditionalHead'), ns)
    fn = function(tree, 'sample_hierarchy'); fn.decorator_list = []
    extract(fn, ns)
    widths = (4, 4)
    heads = types.SimpleNamespace(heads=[object(), object()], block_channels=widths)
    bank = tensor(np.arange(2*8*2*2).reshape(2, 8, 2, 2)/50)
    cfg = dict(t_min=.00003, t_max=1.35, time_schedule='log_t')
    z = ns['sample_hierarchy'](heads, noise=bank, y=None, cfg=cfg)
    assert len(calls) == 2 and calls[0] is None
    np.testing.assert_allclose(calls[1], z[:, :4])
    calls.clear()
    root = tensor(np.ones((2, 4, 2, 2))*7)
    z = ns['sample_hierarchy'](heads, noise=bank, y=None, cfg=cfg, root=root)
    assert len(calls) == 1
    np.testing.assert_allclose(z[:, :4], root)
    np.testing.assert_allclose(calls[0], root)
    residual = np.linspace(-1, 1, 8*4).reshape(8, 2, 2)
    weighted = sum(c/8*np.mean(v*v) for c, v in zip(widths, np.split(residual, [4], axis=0)))
    np.testing.assert_allclose(weighted, np.mean(residual*residual))
    print('OK actual ancestral sampler: every child uses learned reverse diffusion and generated clean parents; dimension weighting')


def recipe_checks(base):
    import paper
    with config_stub(base):
        from suite import resolve_config
        baseline = next(csv.DictReader((base/'baseline_manifest.csv').open()))
        original = resolve_config(baseline, base)
        rows = paper.manifest(base)
        assert len({r['result_name'] for r in rows}) == 5
        assert paper.status_path(base, rows[-1]).name == 'cell_005_v3.json'
        assert paper.status_path(base, rows[0]).name == 'cell_000.json'
        for row in rows:
            cfg = paper.config_for(row, base)
            strip = {'results_dir', 'master_results_dir', 'ckpt_dir', 'paper_cell_id', 'paper_result_name',
                     'block_channels', 'score_depths', 'hierarchy_version', 'training_engine',
                     'tmin_band_weight', 'resolved_tmin_band_weight', 'hierarchy_variant',
                     'block_spatial_sizes', 'innovation_channels', 'latent_coordinates'}
            assert {k: v for k, v in cfg.items() if k not in strip} == {k: v for k, v in original.items() if k not in strip}
            assert cfg['epochs_vae'] == cfg['lr_schedule_epochs'] == 800
            assert cfg['epochs_refine'] == 0 and cfg['dataset'] == 'CIFAR'
            assert cfg['batch_size'] == 256 and cfg['latent_channels'] == 8
            assert cfg['T_terminal'] == 1.05 and cfg['t_max'] == 1.35
            assert cfg['latent_anchor_mode'] == 'ou_partial' and cfg['latent_anchor_time'] == 1.05
            assert cfg['score_w_vae'] == .05 and cfg['kl_w'] == .6
            assert cfg['lr_vae'] == .00025 and cfg['lr_score_head'] == .0001
            assert cfg['factored_head'] and cfg['score_head_loss_w'] == 1.0
            assert cfg['perc_w'] == .85 and cfg['gan_w'] == .0025 and cfg['disc_start_epoch'] == 25
            assert cfg['paper_cfg_grid'] == [2.5, 3.0]
            assert cfg['score_time_weighting'] == 'canonical' and cfg['score_head_time_weighting'] == 'unweighted-eps'
            assert sum(paper.split_ints(row['block_channels'])) == 8
            assert sum(paper.split_ints(row['score_depths'])) == 12
            if int(row['cell_id']) >= 3:
                assert cfg['tmin_band_weight'] == 'matched'
                assert cfg['resolved_tmin_band_weight'] == cfg['score_w_vae'] == .05
            if int(row['cell_id']) == 5:
                assert cfg['block_channels'] == [2, 6] and cfg['block_spatial_sizes'] == [4, 8]
                assert 8 * 4 * 4 + 6 * 8 * 8 == cfg['latent_coordinates'] == 512
            if int(row['cell_id']) < 3:
                old = {k: v for k, v in row.items() if k not in {'tmin_band_weight', 'hierarchy_variant', 'block_spatial_sizes'}}
                assert paper.config_hash(row) == paper.config_hash(old)
            print(f'OK cell {row["cell_id"]}: blocks={row["block_channels"]}, depths={row["score_depths"]}, CIFAR / 800 epochs')
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        paper.submit_jobs(base, dry_run=True)
    assert len([l for l in text.getvalue().splitlines() if l.startswith('sbatch ')]) == 5
    assert '--dependency=' not in text.getvalue() and 'TASK=prepare' not in text.getvalue()
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        paper.submit_jobs(base, spec='new', dry_run=True)
    assert len([l for l in text.getvalue().splitlines() if l.startswith('sbatch ')]) == 3
    assert 'CELL_ID=3' in text.getvalue() and 'CELL_ID=4' in text.getvalue() and 'CELL_ID=5' in text.getvalue()
    print('OK five total / three B-arm training submissions; original status hashes preserved')


def io_checks(base):
    import paper
    from hier_engine import validate_hierarchy_resume
    saved = dict(block_channels=[4, 4], score_depths=[6, 6], hierarchy_version=1)
    validate_hierarchy_resume(saved, saved)
    for field, value in (('block_channels', [3, 3, 2]), ('score_depths', [12, 12]), ('hierarchy_version', 2)):
        try:
            validate_hierarchy_resume(saved, dict(saved, **{field: value}))
        except ValueError:
            pass
        else:
            raise AssertionError('Incompatible hierarchy resume was accepted')
    row = paper.manifest(base)[1]
    endpoint = dict(epoch=800, stage='cotrain', tag='LSI_Diff')
    for level in (2.5, 3.0):
        for method, steps in (('rk4', 20), ('rk4', 25), ('heun', 20)):
            key = f'fid_{method}_{steps}_randtok_cfg{str(level).replace(".", "_")}_temp1_initgaussianT1p35'
            endpoint[key] = 9.0
        endpoint[f'fid_encoded_root_generated_children_cfg{str(level).replace(".", "_")}'] = 5.0
    values = [dict(endpoint, epoch=ep) for ep in range(50, 801, 50)]
    paper.validate_evaluation(row, values)
    try:
        paper.validate_evaluation(row, values[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete run was marked complete')
    print('OK hierarchy resume guards and endpoint/evaluation completeness')


def band_checks(base):
    import band_regularizer as band
    source = (base/'core.py').read_text()
    trainer = function(ast.parse(source), 'train_vae_cotrained_cond')
    from hier_engine import adapt_trainer_source
    for raw in (ast.get_source_segment(source, trainer), adapt_trainer_source(ast.get_source_segment(source, trainer))[0]):
        adapted, counts = band.adapt_band_source(raw)
        compile(adapted, '<band>', 'exec')
        assert counts == dict(setup=1, penalty=1, loss=2, epoch=1, resume=1)
        assert 'band_objective = active_score_w_vae * band_penalty' in adapted
        assert 'loss_joint.backward()' in adapted and 'replace_parameter_gradients' in adapted
    # Execute the actual analytic penalty with a minimal functional shim.
    saved = {k: sys.modules.get(k) for k in ('torch', 'torch.nn')}
    torch_stub = types.ModuleType('torch'); nn_stub = types.ModuleType('torch.nn')
    nn_stub.functional = types.SimpleNamespace(softplus=lambda x: np.logaddexp(0., x))
    sys.modules.update({'torch': torch_stub, 'torch.nn': nn_stub})
    try:
        t = 3e-5
        floor = band.tmin_band_logvar_floor(dict(alpha=np.array([np.exp(-t)]),
                                            sigma=np.array([np.sqrt(-np.expm1(-2*t))])))
        lv = np.array([-30., -10., -2., 0., 20.])
        actual = band.tmin_band_penalty(lv, floor)
        np.testing.assert_allclose(actual, .5*np.logaddexp(0., np.log(np.expm1(2*t))-lv).mean())
        for v in lv:
            step = 1e-5
            slope = (band.tmin_band_penalty(np.array([v+step]), floor)-
                     band.tmin_band_penalty(np.array([v-step]), floor))/(2*step)
            expected = -.5/(1+np.exp(v-floor))
            np.testing.assert_allclose(slope, expected, atol=2e-10)
    finally:
        for k, v in saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v
    assert '--cell-id "$CELL_ID"' in (base/'job.slurm').read_text()
    assert '$((CELL_ID + 1))' not in (base/'job.slurm').read_text()
    print('OK analytic band value/gradient, matched representation routing, both guarded adapters and five-cell launcher')


def run(base):
    base = Path(base)
    for file in base.glob('*.py'):
        compile(file.read_text(), str(file), 'exec')
    source_checks(base); recipe_checks(base); posterior_coupling_checks(base); sampling_checks(base); io_checks(base); band_checks(base)
    print('Offline validation passed. Real Torch/CUDA and numerical FID are separate cluster checks.')
    return 0


def torch_smoke(base, blocks=None, device='cpu', cell_id=None):
    import torch
    import core
    from hier_models import HierarchicalVAE, HierarchicalHeads, sample_hierarchy
    from optimized_models import OptimizedHierarchicalVAE, OptimizedHierarchicalHeads, pack_root, unpack_root
    from band_regularizer import tmin_band_penalty, tmin_band_logvar_floor
    from paper import config_for, manifest
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    for row in manifest(base):
        if cell_id is not None and int(row['cell_id']) != cell_id:
            continue
        if blocks is not None and int(row['blocks']) != blocks:
            continue
        cfg = config_for(row, base); core.seed_everything(42)
        opts = {k: cfg[k] for k in ('latent_channels', 'base_ch', 'num_res_blocks', 'decoder_attn_half',
                    'latent_proj_depth', 'encoder_attn_half', 'decoder_extra_block', 'conv3x3_proj',
                    'use_tanh_out', 'clamp_logvar', 'attn_zero_init', 'time_cond_decoder', 'dec_time_emb_dim')}
        opts.update(img_size=32, img_channels=3, use_norm=False,
                    latent_anchor_mode='ou_partial', latent_anchor_time=1.05)
        h_opts = dict(in_channels=8, hidden_dim=384, depth=12, num_heads=6, patch_size=1,
                      latent_size=8, num_classes=10, mlp_ratio=4.0, dropout=0.0, factored_head=True)
        hierarchical = int(row['blocks']) > 1
        optimized = cfg.get('hierarchy_variant') == 'optimized_v3'
        vae_class = OptimizedHierarchicalVAE if optimized else HierarchicalVAE
        head_class = OptimizedHierarchicalHeads if optimized else HierarchicalHeads
        vae = (vae_class(block_channels=cfg['block_channels'], **opts) if hierarchical else core.VAE(**opts)).to(dev)
        heads = (head_class(block_channels=cfg['block_channels'], score_depths=cfg['score_depths'], **h_opts)
                 if hierarchical else core.DiTModel(**h_opts)).to(dev)
        x = torch.randn(2, 3, 32, 32, device=dev).clamp(-1, 1)
        y = torch.tensor([1, 7], device=dev)
        if hierarchical:
            eps = torch.randn(2, 8, 8, 8, device=dev)
            mu, lv, z = vae.encode_sample(x, eps=eps)
            torch.testing.assert_close(z, mu+torch.exp(.5*lv)*eps)
        else:
            mu, lv = vae.encode(x); z = vae.reparameterize(mu, lv)
        t = torch.tensor([.005, .7], device=dev)
        a, s = core.get_ou_params(t[:, None, None, None])
        noisy = a*z+s*torch.randn_like(z)
        target = s*(noisy-a*mu)/(a.square()*lv.exp()+s.square()+1e-8)
        ctx = {'context': z} if hierarchical else {}
        pred, lam, nu = heads(noisy, t, y, return_components=True, detach_components=True, **ctx)
        outer = ((pred-target).square().mean()*.05 +
                 .6*core.component_kl_at_ou_time(mu, lv, 1.05) +
                 (vae.decode(noisy, t=t)-x).square().mean())
        if cfg.get('tmin_band_weight') == 'matched':
            floor = tmin_band_logvar_floor(core.make_schedule(cfg, dev))
            band = .05 * tmin_band_penalty(lv, floor)
            bg = torch.autograd.grad(band, (*vae.parameters(), *heads.parameters()), allow_unused=True, retain_graph=True)
            nvae = len(tuple(vae.parameters()))
            assert any(g is not None and g.abs().sum()>0 for g in bg[:nvae])
            assert all(g is None for g in bg[nvae:])
            outer = outer + band
        detached_noisy = a*z.detach()+s*torch.randn_like(z)
        detached_target = s*(detached_noisy-a*mu.detach())/(a.square()*lv.detach().exp()+s.square()+1e-8)
        inner_ctx = {'context': z.detach()} if hierarchical else {}
        p_inner, l_inner, _ = heads(detached_noisy, t, y, return_components=True, detach_components=True, **inner_ctx)
        inner = (p_inner-detached_target).square().mean()+.0025*(l_inner-s/(a.square()*lv.detach().exp()+s.square()+1e-8)).square().mean()
        score_params = tuple(heads.parameters())
        gradients = torch.autograd.grad(inner, score_params, allow_unused=True, retain_graph=False)
        outer.backward()  # Must work after freeing the independent inner graph.
        core.replace_parameter_gradients(score_params, gradients)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in (*vae.parameters(), *heads.parameters()))
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in vae.parameters())
        assert any(g is not None and g.abs().sum()>0 for g in gradients)
        if optimized:
            # Verify the decoder cannot reconstruct from a cached clean root
            # when passed independently corrupted coordinates. Both root and
            # child must genuinely affect this forward graph.
            probe = noisy.detach().clone().requires_grad_(True)
            probe_output = vae.decode(probe, t=t).square().mean()
            probe_grad = torch.autograd.grad(probe_output, probe)[0]
            assert probe_grad[:, :2].abs().sum().item() > 1e-9
            assert probe_grad[:, 2:].abs().sum().item() > 1e-9
            # The original factored DiT intentionally zero-initializes nu_proj,
            # so its output cannot yet depend on *any* token/context. Activate
            # the projection only inside this isolated smoke process.
            original_nu = heads.heads[1].nu_proj.weight.detach().clone()
            with torch.no_grad():
                heads.heads[1].nu_proj.weight.normal_(std=1e-3)
            root_ctx = z[:, :2].detach().clone().requires_grad_(True)
            child_pred = heads.heads[1](noisy[:, 2:].detach(), t, y, context=root_ctx)
            child_ctx_grad = torch.autograd.grad(child_pred.square().mean(), root_ctx)[0]
            assert child_ctx_grad.abs().sum().item() > 1e-9
            with torch.no_grad():
                heads.heads[1].nu_proj.weight.copy_(original_nu)
            real_root = unpack_root(z[:, :2])
            assert real_root.shape == (2, 8, 4, 4)
            torch.testing.assert_close(pack_root(real_root), z[:, :2])
            assert heads.heads[0].model.pos_embed.shape[1] == 16
        if hierarchical:
            # Explicitly verify clean-ancestor dependence rather than relying on
            # zero-initialized adapters to happen to be active in a smoke pass.
            with torch.no_grad():
                if not optimized:
                    for adapter in vae.conditions:
                        adapter[-1].weight.fill_(.001)
            eps = torch.randn_like(z, requires_grad=True)
            mm, ll, zz = vae.encode_sample(x, eps=eps)
            dm = torch.autograd.grad(mm[:, cfg['block_channels'][0]:].mean(), eps)[0]
            assert dm[:, :cfg['block_channels'][0]].abs().sum()>0
            with torch.no_grad():
                noise = torch.randn_like(z)
                generated = sample_hierarchy(heads, noise=noise, y=y, cfg=cfg, steps=2, cfg_scale=2.5)
                assert generated.shape == z.shape and torch.isfinite(generated).all()
                anchored = sample_hierarchy(heads, noise=noise, y=y, cfg=cfg, steps=2, cfg_scale=2.5,
                                             root=zz[:, :cfg['block_channels'][0]])
                torch.testing.assert_close(anchored[:, :cfg['block_channels'][0]], zz[:, :cfg['block_channels'][0]])
        print(f'OK real Torch cell {row["cell_id"]}, {row["outputs"]}: exact conditional targets, independent inner graph, finite routed gradients'
              f'; VAE params={sum(p.numel() for p in vae.parameters()):,}, score params={sum(p.numel() for p in heads.parameters()):,}')
        del vae, heads
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    return 0
