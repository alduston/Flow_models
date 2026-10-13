"""Eight controlled 500-epoch arms; all scientific settings live in manifest.csv."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

SCIENTIFIC_FILES = ('suite.py', 'core.py', 'train_hier_csem_fmnist.py',
                    'fmnist_reference.py', 'paper_io.py')


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
    mapping = {'mode':'mode', 'z1_channels':'z1-channels', 'z2_channels':'z2-channels',
        'root_size':'root-size', 'child_posterior':'child-posterior', 'score_depths':'score-depths',
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
    values += ['--output', str(Path(base)/'results'/row['result_name']),
               '--data-root', str(Path(data_root).resolve() if data_root else Path(base)/'data_cache_fmnist'),
               '--device', 'cuda']
    return values


def config_for(row, base, data_root=None):
    from train_hier_csem_fmnist import parse_args
    return parse_args(cli_args(row, base, data_root))


def validate_manifest(rows):
    if [int(r['cell_id']) for r in rows] != list(range(8)):
        raise ValueError('Expected exactly eight cells 0..7')
    if len({r['result_name'] for r in rows}) != 8:
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
    for a,b in ((0,2),(1,3)):
        omit = {'cell_id','seed','result_name','outputs','notes'}
        if {k:v for k,v in rows[a].items() if k not in omit} != {k:v for k,v in rows[b].items() if k not in omit}:
            raise ValueError('Repeat-seed arms must be scientifically identical')
    return True
