"""Pure sampler/metric contracts and locked, staged dataset preparation."""
from pathlib import Path
from contextlib import contextmanager
import fcntl
import json
import os
import re
import shutil
import tempfile
import time


def save_csv_atomic(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


@contextmanager
def dataset_lock(cache, key, exclusive=True):
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ('.' + key + '.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield cache / key
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _open_dataset_pair(cls, root, transform=None, download=False, kwargs=None):
    kw = dict(kwargs or {})
    pair = tuple(cls(str(root), train=train, download=download, transform=transform, **kw)
                 for train in (True, False))
    for dataset in pair:
        if not len(dataset):
            raise RuntimeError('Empty dataset')
        check = getattr(dataset, '_check_integrity', None)
        if check is not None and not check():
            raise RuntimeError('Dataset integrity check failed')
    return pair


def prepare_dataset(cls, cache, key, kwargs=None, attempts=3):
    """Validate both splits; download to fresh staging, then publish under lock.

    Never extract into a live cache. Failed staging directories are removed;
    a corrupt prior snapshot is retained in quarantine, not silently reused.
    """
    with dataset_lock(cache, key) as destination:
        try:
            _open_dataset_pair(cls, destination, kwargs=kwargs)
            return destination
        except Exception as exc:
            reason = repr(exc)
        errors = []
        for attempt in range(attempts):
            stage = Path(tempfile.mkdtemp(prefix='.' + key + '-stage-', dir=destination.parent))
            try:
                _open_dataset_pair(cls, stage, download=True, kwargs=kwargs)
                _open_dataset_pair(cls, stage, download=False, kwargs=kwargs)
                (stage / 'prepared.json').write_text(json.dumps({'dataset': key, 'validated': True}) + '\n')
                if destination.exists():
                    quarantine = destination.with_name('.' + key + '-corrupt-' + str(time.time_ns()))
                    os.replace(destination, quarantine)
                    (quarantine / 'quarantine_reason.txt').write_text(reason + '\n')
                os.replace(stage, destination)
                return destination
            except Exception as exc:
                errors.append(repr(exc))
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        raise RuntimeError(f'Could not prepare {key} after {attempts} clean attempts: {errors}')


def load_prepared_pair(cls, cache, key, transform=None, kwargs=None):
    with dataset_lock(cache, key, exclusive=False) as destination:
        try:
            return _open_dataset_pair(cls, destination, transform=transform, kwargs=kwargs)
        except Exception as exc:
            raise RuntimeError(f'{key} cache is missing/corrupt. Run paper.py prepare-data before training; '
                               'training jobs never download or extract datasets.') from exc


def normalize_metrics(frame, *, recover_v5=False):
    """Preserve CFG labels and reject ambiguous/control-character headers.

    V5's documented Aug13 insertion order was CFG3 first, CFG1.5 second.
    Pandas mangles the second duplicate as '.1'. Recovery is opt-in and the
    importer verifies the dataset/mode before invoking it.
    """
    mapping = {}
    for column in frame.columns:
        name = str(column)
        if '\x01' in name:
            if not recover_v5:
                raise ValueError('Broken V5 metric header; use import-v5 for explicit recovery')
            m = re.fullmatch(r'(.*)\x01(_temp1_initgaussianT2)(\.1)?', name)
            original_heun50 = bool(m and re.fullmatch(r'.*_heun_50', m.group(1)))
            if not m or (not m.group(3) and name + '.1' not in frame.columns and not original_heun50):
                raise ValueError(f'Ambiguous V5 CFG recovery: {name!r}')
            name = m.group(1) + ('_randtok_cfg1_5' if m.group(3) else '_randtok_cfg3_0') + m.group(2)
        if '_randtok_cfg' in name and '_init' not in name:
            name = re.sub(r'(_randtok_cfg[0-9]+_[0-9]+)',
                          lambda m: m.group(1) + '_temp1_initgaussianT2', name)
        if any(ord(c) < 32 for c in name):
            raise ValueError(f'Control character in metric header: {name!r}')
        mapping[column] = name
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Metric normalization would create duplicate columns')
    return frame.rename(columns=mapping), {str(k): v for k, v in mapping.items() if k != v}


def validate_resume_config(saved, current):
    keys = ['dataset', 'seed', 'T_terminal', 't_max', 'latent_anchor_mode', 'lr_schedule_epochs',
            'score_w_vae', 'kl_w', 'lr_score_head', 'lr_vae', 'lr_refine', 'epochs_vae',
            'epochs_refine', 'freeze_score_in_cotrain', 'train_tracking_head',
            'score_time_weighting', 'score_head_time_weighting', 'cotrain_head']
    for key in keys:
        if saved.get(key) != current.get(key):
            raise ValueError(f'Resume configuration mismatch: {key}')


def sampler_configs(cfg, has_model=True):
    """The same plan determines which oracle banks are actually required."""
    configs = []
    T = float(cfg['t_max'])
    g = float(cfg.get('cfg_eval_scale', 3.0))
    if cfg.get('deployment_only', False):
        for steps in cfg.get('deployment_rk4_step_grid', [25]):
            for level in cfg.get('deployment_cfg_grid', [g]):
                for temp in cfg.get('deployment_temperature_grid', [1.0]):
                    configs.append(dict(method='rk4_ode', steps=int(steps), desc='Deployment Gaussian',
                        use_rand_token=True, cfg_level=float(level), init_mode='gaussian-T',
                        init_temperature=float(temp), sampler_t_max=T))
    else:
        configs.append(dict(method='VAE_Rec_eps', steps=0, desc='Recon (posterior z)',
                            use_rand_token=False, init_mode='reconstruction'))
        if has_model and not cfg.get('oracle_nfe_eval_only', False):
            oracle = bool(cfg.get('eval_tk_vs_t_comparison', True)) or cfg.get('eval_sampling_init', 'gaussian') != 'gaussian'
            TK = float(cfg['T_terminal'])
            horizons = [('T', T, int(cfg.get('eval_rk4_steps_t', 25)))]
            if TK > float(cfg['t_min']):
                horizons.insert(0, ('TK', TK, int(cfg.get('eval_rk4_steps_tk', 25))))
            for label, horizon, steps in horizons:
                if oracle:
                    init = 'oracle-qtk-class' if label == 'TK' else 'oracle-qT-class'
                    configs.append(dict(method='rk4_ode', steps=steps, desc='Oracle initialization',
                        use_rand_token=True, cfg_level=g, init_mode=init, sampler_t_max=horizon))
                configs.append(dict(method='rk4_ode', steps=steps, desc='Gaussian initialization',
                    use_rand_token=True, cfg_level=g, init_mode='gaussian-' + label,
                    init_temperature=1.0, sampler_t_max=horizon))
            if cfg.get('paper_solver_suite', False):
                for method in ['heun_sde', 'rk4_ode']:
                    configs.append(dict(method=method, steps=int(cfg.get('paper_solver_steps', 20)),
                        desc='Paper solver comparison', use_rand_token=True, cfg_level=g,
                        init_mode='gaussian-T', init_temperature=1.0, sampler_t_max=T))
    if cfg.get('paper_cfg_grid') and has_model:
        for level in dict.fromkeys(float(x) for x in cfg['paper_cfg_grid']):
            for method, steps in [('rk4_ode', 20), ('rk4_ode', 25), ('heun_sde', 20)]:
                if not any(c.get('method') == method and c.get('steps') == steps and
                           c.get('cfg_level') == level and c.get('init_mode') == 'gaussian-T' for c in configs):
                    configs.append(dict(method=method, steps=steps, desc='Paper Gaussian comparison',
                        use_rand_token=True, cfg_level=level, init_mode='gaussian-T',
                        init_temperature=1.0, sampler_t_max=T))
    return configs
