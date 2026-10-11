"""Small, audited adapter of the shipped CIFAR trainer; flat bypasses this file.

Keep the original time samplers, losses, auxiliary precision loss, optimizers,
gradient replacement, joint clipping, EMA, checkpoints and resume logic.
Change only model construction, sequential posterior sampling, explicit clean
context, hierarchy diagnostics and the ancestral evaluator. Source-shape guards
refuse to adapt an engine different from the pinned one.
"""
from __future__ import annotations
import ast
import inspect


def adapt_trainer_source(source):
    tree = ast.parse(source)
    counts = {'vae': 0, 'heads': 0, 'encode': 0, 'resample': 0, 'context': 0,
              'base_encode': 0, 'metrics': 0, 'batch': 0, 'epoch': 0, 'resume': 0, 'parameter_counts': 0}

    class Adapter(ast.NodeTransformer):
        in_joint = False
        def visit_Call(self, node):
            self.generic_visit(node)
            if not isinstance(node.func, ast.Name):
                return node
            name = node.func.id
            if name == 'VAE':
                node.func.id = 'HierarchicalVAE'
                node.keywords.append(ast.keyword(arg='block_channels', value=ast.parse("cfg['block_channels']", mode='eval').body))
                counts['vae'] += 1
            elif name == 'UNetModel':
                node.func.id = 'HierarchicalHeads'
                node.keywords.extend([
                    ast.keyword(arg='block_channels', value=ast.parse("cfg['block_channels']", mode='eval').body),
                    ast.keyword(arg='score_depths', value=ast.parse("cfg['score_depths']", mode='eval').body),
                ])
                counts['heads'] += 1
            elif name in {'unet_lsi', 'unet_control'}:
                outer = ast.unparse(node.args[0]) in {'z_t_for_score', 'z_mu_t_for_score'}
                value = ast.parse('z0' if outer else 'z0.detach()', mode='eval').body
                node.keywords.append(ast.keyword(arg='context', value=value))
                counts['context'] += 1
            return node

        def visit_Assign(self, node):
            if len(node.targets) == 1:
                target = ast.unparse(node.targets[0])
                value = ast.unparse(node.value)
                if target == '(mu_base, logvar_base)' and value == 'vae.encode(x)':
                    counts['base_encode'] += 1
                    return None
                if target == '(mu, logvar)' and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'vae.encode':
                    counts['encode'] += 1
                    return ast.parse('mu, logvar, z0 = vae.encode_sample(x)').body[0]
                if target == 'z0' and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'vae.reparameterize':
                    counts['resample'] += 1
                    return None
                if target == 'epoch_metrics' and self.in_joint:
                    counts['metrics'] += 1
                    return [self.generic_visit(node), *ast.parse(
                        "epoch_metrics.update({k: v/n_batches for k, v in block_metric_sums.items()})"
                    ).body]
                if target == 'ema_decay':
                    counts['parameter_counts'] += 1
                    return [self.generic_visit(node), *ast.parse(
                        "cfg['model_parameter_counts'] = {"
                        "'vae': sum(p.numel() for p in vae.parameters()), "
                        "'score_heads': [sum(p.numel() for p in head.parameters()) for head in unet_lsi.heads], "
                        "'total_score': sum(p.numel() for p in unet_lsi.parameters())}\n"
                        "print('Model parameter counts:', cfg['model_parameter_counts'], flush=True)"
                    ).body]
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'validate_resume_config':
                counts['resume'] += 1
                return [node, *ast.parse("validate_hierarchy_resume(state['cfg'], cfg)").body]
            return self.generic_visit(node)

        def visit_For(self, node):
            target = ast.unparse(node.target)
            joint = target == 'ep' and ast.unparse(node.iter) == "range(start_epoch, cfg['epochs_vae'])"
            previous = self.in_joint
            self.in_joint = previous or joint
            self.generic_visit(node)
            if joint:
                counts['epoch'] += 1
                node.body.insert(0, ast.parse('block_metric_sums = {}').body[0])
            if self.in_joint and target == '(x, y)' and isinstance(node.iter, ast.Call) and ast.unparse(node.iter.func) == 'tqdm':
                counts['batch'] += 1
                node.body.extend(ast.parse(
                    'accumulate_block_metrics(block_metric_sums, cfg, mu, logvar, eps_pred_lsi, '
                    'eps_target_lsi, score_time_weights, kl_sched)'
                ).body)
            self.in_joint = previous
            return node

    tree = ast.fix_missing_locations(Adapter().visit(tree))
    expected = dict(vae=1, heads=4, encode=2, resample=2, context=17,
                    base_encode=1, metrics=1, batch=1, epoch=1, resume=1, parameter_counts=1)
    if counts != expected:
        raise RuntimeError(f'Pinned trainer shape changed: {counts} != {expected}')
    return ast.unparse(tree), counts


def validate_hierarchy_resume(saved, current):
    for key in ('block_channels', 'score_depths', 'hierarchy_version'):
        if saved.get(key) != current.get(key):
            raise ValueError(f'Resume configuration mismatch: {key}')


def accumulate_block_metrics(totals, cfg, mu, logvar, prediction, target, weights, kl_schedule):
    import torch
    import core
    with torch.no_grad():
        parts = [v.split(cfg['block_channels'], dim=1) for v in (mu, logvar, prediction, target)]
        for i, (m, lv, p, c) in enumerate(zip(*parts), 1):
            values = {
                f'score_{i}': ((p-c).square().flatten(1).mean(1)*weights).mean().item(),
                f'kl_{i}': core.terminal_component_kl_from_schedule(m, lv, kl_schedule).item(),
                f'var_{i}': lv.exp().mean().item(),
            }
            for key, val in values.items():
                totals[key] = totals.get(key, 0.0) + val


def train_hierarchy(cfg):
    import core
    from hier_models import HierarchicalVAE, HierarchicalHeads
    from hier_eval import evaluate_hierarchy
    if len(cfg['block_channels']) not in (2, 3) or cfg['epochs_refine'] != 0:
        raise ValueError('The hierarchy adapter supports only 2/3-block cotraining with no refinement')
    source, counts = adapt_trainer_source(inspect.getsource(core.train_vae_cotrained_cond))
    namespace = dict(core.__dict__)
    namespace.update(HierarchicalVAE=HierarchicalVAE, HierarchicalHeads=HierarchicalHeads,
                     evaluate_current_state=evaluate_hierarchy,
                     accumulate_block_metrics=accumulate_block_metrics,
                     validate_hierarchy_resume=validate_hierarchy_resume)
    exec(compile(source, '<pinned_cifar_hierarchy_trainer>', 'exec'), namespace)
    cfg['hierarchy_adapter_counts'] = counts
    return namespace['train_vae_cotrained_cond'](cfg)
