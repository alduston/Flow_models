"""Five controlled 500-epoch arms; all scientific settings live in manifest.csv."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

SCIENTIFIC_FILES = ('suite.py', 'core.py', 'train_hier_csem_fmnist.py',
                    'fmnist_reference.py', 'paper_io.py', 'train_hier_csem_fmnist_3block.py', 'diagnostics.py')


def manifest(base):
    with (Path(base)/'manifest.csv').open(newline='') as f:
        return list(csv.DictReader(f))


def source_hashes(base):
    return {name: hashlib.sha256((Path(base)/name).read_bytes()).hexdigest()
            for name in SCIENTIFIC_FILES}


def config_hash(row, base):
    return hashlib.sha256(json.dumps({'row': row, 'source': source_hashes(base)},
                                    sort_keys=True).encode()).hexdigest()


def cli_args(row, base, data_root=None):
    mapping = {'eval_risk_samples':'eval-risk-samples', 'eval_risk_bank_sizes':'eval-risk-bank-sizes',
        'eval_risk_time_bins':'eval-risk-time-bins', 'eval_block_cfg_samples':'eval-block-cfg-samples',
        'eval_block_cfg_scales':'eval-block-cfg-scales', 'mode':'mode', 'z1_channels':'z1-channels', 'z2_channels':'z2-channels',
        
        'epochs_joint':'epochs', 'seed':'seed', 'batch':'batch', 'num_workers':'num-workers',
        'lr_vae':'lr-vae', 'lr_head':'lr-head', 'score_weight':'score-weight',
        'csem_rep_weight':'csem-rep-weight', 'terminal_weight':'terminal-weight',
        'tmin_band_weight':'tmin-band-weight', 'recon_root':'recon-root',
        'perc_weight':'perc-weight', 'gan_weight':'gan-weight', 'disc_start':'disc-start',
        'lr_disc':'lr-disc', 'label_dropout':'label-dropout', 'ema':'ema',
        't_min':'t-min', 't_max':'t-max', 'decode_time':'decode-time',
        'eval_every':'eval-every', 'eval_samples':'eval-samples', 'eval_batch':'eval-batch',
        'eval_check_samples':'eval-check-samples', 'eval_usage_samples':'eval-usage-samples',
        'eval_grad_batch':'eval-grad-batch', 'rk4_steps':'rk4-steps', 'cfg':'cfg',
        'fid_backend':'fid-backend', 'feature_batch':'feature-batch',
        'save_array_samples':'save-array-samples', 'checkpoint_every':'checkpoint-every'}
    values = []
    for key, flag in mapping.items():
        values += ['--'+flag, str(row[key])]
    if int(row['blocks']) == 3:
        values += ['--z3-channels', row['z3_channels'], '--conditioning', row['conditioning']]
    values += ['--output', str(Path(base)/'results'/row['result_name']),
               '--data-root', str(Path(data_root).resolve() if data_root else Path(base)/'data_cache_fmnist'),
               '--device', 'cuda']
    return values


def config_for(row, base, data_root=None):
    import importlib
    engine = importlib.import_module('train_hier_csem_fmnist_3block' if int(row['blocks']) == 3 else 'train_hier_csem_fmnist')
    args = engine.parse_args(cli_args(row, base, data_root))
    args.blocks = int(row['blocks'])
    return args


def validate_manifest(rows):
    if [int(r['cell_id']) for r in rows] != list(range(5)):
        raise ValueError('Expected exactly five cells 0..4')
    if len({r['result_name'] for r in rows}) != 5:
        raise ValueError('Result names must be unique')
    common = ('epochs_joint','batch','lr_vae','lr_head','score_weight','csem_rep_weight',
              'terminal_weight','tmin_band_weight','recon_root','perc_weight','gan_weight',
              'disc_start','lr_disc','label_dropout','ema','t_min','t_max','decode_time',
              'eval_every','eval_samples','cfg','rk4_steps','fid_backend')
    for key in common:
        if len({r[key] for r in rows}) != 1:
            raise ValueError(f'Unplanned recipe variation: {key}')
    for r in rows:
        flat = r['mode'] == 'flat'
        dims = [256] if flat else [int(r['z1_channels'])*int(r['root_size'])**2,
                                  int(r['z2_channels'])*64]
        if int(r['blocks']) == 3: dims.append(int(r['z3_channels'])*64)
        if sum(dims) != 256 or ':'.join(map(str,dims)) != r['block_coordinates']:
            raise ValueError(f'Coordinate budget mismatch: cell {r["cell_id"]}')
        depths = list(map(int, r['score_depths'].split(':')))
        if len(depths) != len(dims) or min(depths) < 1:
            raise ValueError('Need one positive score depth per block')
        if int(r['epochs_joint']) != 500 or r['recon_root'] != 'noisy' or r['tmin_band_weight'] != 'auto':
            raise ValueError('All arms require 500 epochs, noisy-root reconstruction, automatic B')
        if r['fid_backend'] != 'inception-legacy' or int(r['eval_samples']) != 10000:
            raise ValueError('All arms require the same 10000-sample legacy FID')
        if '/' in r['result_name'] or '..' in r['result_name']:
            raise ValueError('Result names must be local directory names')
    expected = [(1,'none'),(2,'full'),(3,'full'),(3,'markov'),(3,'siblings')]
    if [(int(r['blocks']),r['conditioning']) for r in rows] != expected:
        raise ValueError('Expected one flat, one two-block and the three prior DAGs')
    omit = {'cell_id','variant','conditioning','result_name','outputs','notes'}
    full = {k:v for k,v in rows[2].items() if k not in omit}
    for row in rows[3:]:
        if {k:v for k,v in row.items() if k not in omit} != full:
            raise ValueError('Three-block recipes may differ only in prior topology')
    if any(r['seed'] != '42' for r in rows): raise ValueError('All five jobs use paired seed 42')
    return True
