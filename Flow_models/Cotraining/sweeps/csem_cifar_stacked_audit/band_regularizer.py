"""Analytic lower-boundary correction on the encoder representation route.

The original core is unchanged. A guarded AST adapter adds B' only to the new
cells, before the original backward pass and joint clipping. Coordinate means
match the existing CSEM/KL normalization, including unequal block sizes.
"""
from __future__ import annotations
import ast
import inspect
import math


def tmin_band_logvar_floor(schedule):
    a = float(schedule['alpha'][0])
    s = float(schedule['sigma'][0])
    if not (math.isfinite(a) and math.isfinite(s) and a > 0 and s > 0):
        raise ValueError('Band correction requires positive finite cutoff alpha and sigma')
    return 2.0 * (math.log(s) - math.log(a))


def tmin_band_penalty(logvar, floor):
    """B' = 1/2 mean_k log(1 + sigma_min^2/(alpha_min^2 v_k)).

    This is the current note's nonnegative upper bound on information lost
    below t_min, using conditional DIRECT-child variances. No additive constant
    is required. Canonical representation CSEM gives the matching coefficient.
    """
    from torch.nn import functional as F
    return 0.5 * F.softplus(floor - logvar).mean()


def configure_band(cfg, schedule, csem_weight):
    if cfg.get('tmin_band_weight') != 'matched':
        raise ValueError('This adapter requires tmin_band_weight=matched')
    if cfg.get('score_time_weighting') != 'canonical' or schedule is None:
        raise ValueError('Matched band correction requires canonical representation weighting')
    if cfg.get('freeze_score_in_cotrain', False) or cfg.get('cotrain_head', 'lsi') != 'lsi':
        raise ValueError('Band sweep requires active CSEM cotraining')
    floor = tmin_band_logvar_floor(schedule)
    cfg['resolved_tmin_band_weight'] = float(csem_weight)
    cfg['resolved_tmin_band_logvar_floor'] = floor
    cfg['resolved_tmin_band_std_floor'] = math.exp(0.5 * floor)
    cfg['tmin_band_normalization'] = 'mean over all latent coordinates, matching representation CSEM'
    print(f'--> B_tmin: weight={csem_weight:g} (matched CSEM), '
          f'std threshold={cfg["resolved_tmin_band_std_floor"]:.7g}', flush=True)
    return floor


def validate_band_resume(saved, current):
    for key in ('tmin_band_weight', 't_min', 'score_time_weighting'):
        if saved.get(key) != current.get(key):
            raise ValueError(f'Resume configuration mismatch: {key}')


def accumulate_band_metrics(metrics, cfg, logvar, floor, penalty, objective):
    import torch
    with torch.no_grad():
        values = {'band': penalty.item(), 'band_objective': objective.item()}
        widths = cfg.get('block_channels', [int(cfg['latent_channels'])])
        for i, lv in enumerate(logvar.detach().split(widths, dim=1), 1):
            values[f'band_{i}'] = tmin_band_penalty(lv, floor).item()
            values[f'below_{i}'] = (lv < floor).float().mean().item()
            values[f'deep_{i}'] = (lv < floor - 2.0 * math.log(10.0)).float().mean().item()
        for key, value in values.items():
            metrics[key] = metrics.get(key, 0.0) + value


def adapt_band_source(source):
    counts = dict(setup=0, penalty=0, loss=0, epoch=0, resume=0)

    class Adapter(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                return node
            target = node.targets[0].id
            if target == 'score_w_vae':
                counts['setup'] += 1
                return [node, *ast.parse('band_floor = configure_band(cfg, ou_sched, score_w_vae)').body]
            if target == 'stiffness_objective':
                counts['penalty'] += 1
                return [node, *ast.parse(
                    'band_penalty = tmin_band_penalty(logvar, band_floor)\n'
                    'band_objective = active_score_w_vae * band_penalty\n'
                    'accumulate_band_metrics(metrics, cfg, logvar, band_floor, band_penalty, band_objective)'
                ).body]
            if target == 'loss_joint':
                counts['loss'] += 1
                node.value = ast.BinOp(left=node.value, op=ast.Add(), right=ast.Name(id='band_objective', ctx=ast.Load()))
            if target == 'epoch_metrics' and "metrics['loss'] / n_batches" in ast.unparse(node.value):
                counts['epoch'] += 1
                return [node, *ast.parse(
                    "epoch_metrics.update({k: v/n_batches for k, v in metrics.items() "
                    "if k.startswith(('band', 'below_', 'deep_'))})\n"
                    "epoch_metrics['band_weight'] = float(active_score_w_vae)"
                ).body]
            return node

        def visit_Expr(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'validate_resume_config':
                counts['resume'] += 1
                return [node, *ast.parse("validate_band_resume(state['cfg'], cfg)").body]
            return node

    tree = ast.fix_missing_locations(Adapter().visit(ast.parse(source)))
    expected = dict(setup=1, penalty=1, loss=2, epoch=1, resume=1)
    if counts != expected:
        raise RuntimeError(f'Pinned band trainer shape changed: {counts} != {expected}')
    return ast.unparse(tree), counts


def band_namespace():
    return {name: globals()[name] for name in ('configure_band', 'tmin_band_penalty',
            'accumulate_band_metrics', 'validate_band_resume')}


def train_flat_band(cfg):
    import core
    source, counts = adapt_band_source(inspect.getsource(core.train_vae_cotrained_cond))
    namespace = dict(core.__dict__, **band_namespace())
    exec(compile(source, '<pinned_cifar_flat_band_trainer>', 'exec'), namespace)
    cfg['band_adapter_counts'] = counts
    return namespace['train_vae_cotrained_cond'](cfg)
