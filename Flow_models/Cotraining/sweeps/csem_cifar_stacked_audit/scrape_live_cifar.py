#!/usr/bin/env python3
"""Inspect unfinished CIFAR stacked-CSEM jobs. Python standard library only.

Run in the extracted sweep folder:
    python scrape_live_cifar.py
    python scrape_live_cifar.py --logs 'slurm_logs/*.out' --logs-only

Reads a bounded snapshot of each live log, plus existing evaluation CSVs when
available; writes a separate live_compiled directory and a small result ZIP.
It never imports training code, loads checkpoints, or touches a running job.
"""
from __future__ import annotations
import argparse
import codecs
import csv
from datetime import datetime, timezone
import glob
import json
import math
from pathlib import Path
import re
import sys
import zipfile

NUM = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)'
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
HEADER = re.compile(r'---\s*Evaluation:\s*(.+?)\s*@\s*Ep\s*(\d+)\s*---')
SWEEP = re.compile(r'>>>\s*Sweep Results\s*\[(.+?)\]\s*<<<')
CONFIG = re.compile(r'^(VAE_Rec_eps|rk4_ode|heun_sde|ddim)@(\d+)(.*)$')
FID_COLUMN = re.compile(r'^fid_(rk4|heun|ddim)_(\d+)_(.+)$')
CELL = re.compile(r'\bcell(?:_id)?\s*[=:]\s*(\d+)\b')
EPOCH = re.compile(r'^Ep\s+(\d+)\s*\|')
PRODUCTION_TAGS = {'LSI_Diff', 'Frozen'}


def number(value):
    if value is None or str(value).strip() == '':
        return None
    try:
        return float(str(value).strip().lower().replace('p', '.').replace('_', '.'))
    except ValueError:
        return None


def integer(value):
    n = number(value)
    return int(n) if n is not None and math.isfinite(n) and n == int(n) else None


def close(a, b):
    return a is not None and b is not None and math.isclose(a, b, abs_tol=1e-7, rel_tol=1e-7)


def finite(value):
    return value is not None and math.isfinite(value)


def snapshot_lines(path):
    """Read only bytes present at open time. Ignore an unfinished final line."""
    with Path(path).open('rb') as f:
        remaining = Path(path).stat().st_size
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        carry = ''
        lineno = 0
        while remaining:
            chunk = f.read(min(262144, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            text = carry + decoder.decode(chunk)
            parts = re.split(r'[\r\n]', text)
            carry = parts.pop()
            for line in parts:
                lineno += 1
                yield lineno, ANSI.sub('', line).strip()
        # carry has no completed newline; a live writer may still be filling it.


def fallback_manifest():
    result = {}
    arms = [('flat', 1, 'full', 1), ('stack_r0', 2, 'full', 0),
            ('stack_r05', 2, 'full', .5), ('stack_r1', 2, 'full', 1),
            ('stack_r4', 2, 'full', 4), ('stack_decoupled', 2, 'decoupled', 1)]
    for name, blocks, topology, r in arms:
        for seed in (42, 43):
            cid = len(result)
            result[cid] = dict(cell_id=cid, result_name=f'cifar_{name}_s{seed}', seed=seed,
                blocks=blocks, topology=topology, rep_multiplier=r, stage='joint',
                teacher='naive', source='', epochs_joint=800, eval_samples=10000, T_full=1.35)
    for cid in (6, 7, 8, 9):
        for teacher in ('naive', 'corrected', 'dsm'):
            rid = len(result)
            result[rid] = dict(result[cid], cell_id=rid,
                result_name=result[cid]['result_name']+'_frozen_'+teacher,
                stage='frozen', source=cid, teacher=teacher, epochs_joint=50)
    return result


def load_manifest(base, warnings):
    path = base/'manifest.csv'
    if not path.exists():
        warnings.append('No manifest.csv: using the 24-cell design from the delivered CIFAR audit bundle; sample counts are expected, not observed.')
        return fallback_manifest()
    try:
        with path.open(newline='') as f:
            return {int(r['cell_id']): r for r in csv.DictReader(f)}
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        raise SystemExit(f'Cannot read {path}: {exc}')


def metadata(cid, manifest, base):
    raw = manifest.get(cid, {})
    meta = dict(cell_id=cid if cid is not None else '', name=raw.get('result_name', ''),
        seed=integer(raw.get('seed')), blocks=integer(raw.get('blocks')),
        topology=raw.get('topology', ''), r=number(raw.get('rep_multiplier')),
        stage=raw.get('stage', 'unknown'), teacher=raw.get('teacher', ''),
        source_cell=raw.get('source', ''), target_epoch=integer(raw.get('epochs_joint')),
        expected_eval_samples=integer(raw.get('eval_samples')),
        samples_basis='manifest_expected' if raw.get('eval_samples') else 'unknown',
        full_horizon=number(raw.get('T_full')))
    if meta['name']:
        cfg = base/'results'/meta['name']/'paper_cell_config.json'
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text())
                count = integer(data.get('eval_max_samples'))
                if count:
                    meta.update(expected_eval_samples=count, samples_basis='run_config_requested')
                meta['full_horizon'] = number(data.get('t_max')) or meta['full_horizon']
            except (OSError, ValueError, TypeError):
                pass
    return meta


def parse_config(token):
    match = CONFIG.fullmatch(token.strip())
    if not match:
        return None
    method, steps, suffix = match.groups()
    if method == 'VAE_Rec_eps':
        return dict(method='reconstruction', steps=0, cfg=None, temperature=None,
                    init='posterior', horizon=None)
    def extract(pattern):
        m = re.search(pattern, suffix)
        return number(m.group(1)) if m else None
    cfg = extract(r'_cfg([-+]?\d+(?:[._p]\d+)?)')
    temp = extract(r'_temp([-+]?\d+(?:[._p]\d+)?)')
    init_match = re.search(r'_init(gaussianTK|gaussianT|oracleqTK|oracleqT)(\d+(?:[p._]\d+)?)', suffix)
    init, horizon = ('unknown', None)
    if init_match:
        init, horizon = init_match.group(1), number(init_match.group(2))
    if suffix.endswith('_oracle'):
        init += '_oracle_score'
    return dict(method={'rk4_ode': 'rk4', 'heun_sde': 'heun'}.get(method, method),
                steps=int(steps), cfg=cfg, temperature=temp, init=init, horizon=horizon)


def read_log(path, manifest, base, warnings):
    name_match = re.search(r'(?:_c|cell[_-]?)(\d+)(?:_|\.|$)', path.name)
    cid = int(name_match.group(1)) if name_match else None
    epoch = None
    tag = None
    table = False
    metrics = []
    training = []
    epoch_row = None
    headers = []
    errors = []
    modified = path.stat().st_mtime
    for line_no, line in snapshot_lines(path):
        identity = CELL.search(line)
        if identity:
            candidate = int(identity.group(1))
            if cid is not None and cid != candidate:
                warnings.append(f'{path.name}: filename cell {cid} differs from header cell {candidate}; using header.')
            cid = candidate
        header = HEADER.search(line)
        if header:
            tag, epoch = header.group(1).strip(), int(header.group(2))
            table = False
            headers.append((tag, epoch))
            continue
        sweep = SWEEP.search(line)
        if sweep:
            table = bool(tag and sweep.group(1).strip() == tag and epoch is not None)
            continue
        if 'Conditional Sweep Results' in line:
            table = False
        if EPOCH.match(line):
            epoch_row = dict(epoch=int(EPOCH.match(line).group(1)), log_file=str(path), line=line_no, modified=modified)
            for part in line.split('|')[1:]:
                m = re.match(r'\s*([^:]+):\s*('+NUM+r')\s*$', part, re.I)
                if m:
                    epoch_row[m.group(1).strip()] = number(m.group(2))
            training.append(epoch_row)
        elif line.startswith('Stability |') and epoch_row is not None:
            for key, pattern in {
                'raw_score_mse': r'raw/weighted score MSE=('+NUM+r')/',
                'weighted_score_mse': r'raw/weighted score MSE=(?:'+NUM+r')/('+NUM+r')',
                'head_score_mse': r'head-metric MSE=('+NUM+r')',
                'vae_clip_percent': r'clip rate VAE/score=('+NUM+r')%/',
                'head_clip_percent': r'clip rate VAE/score=(?:'+NUM+r')%/('+NUM+r')%',
            }.items():
                m = re.search(pattern, line, re.I)
                if m:
                    epoch_row[key] = number(m.group(1))
        if any(s in line.lower() for s in ('traceback (most recent call last)', 'out of memory', 'slurmstepd: error:')):
            errors.append(f'line {line_no}: {line[:250]}')
        if not table or '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|')]
        parsed = parse_config(parts[0])
        # All nine printed columns must be present: do not trust a half-written row.
        if not parsed or len(parts) < 9:
            continue
        vals = [number(p) for p in parts[2:9]]
        if any(v is None for v in vals):
            continue
        record = dict(epoch=epoch, tag=tag, config=parts[0], description=parts[1], **parsed,
            **dict(zip(('fid', 'kid', 'sw2', 'diversity', 'lsi_gap', 'mse_eps', 'mse_score'), vals)),
            value_source='log_rounded', source_file=str(path), line=line_no, modified=modified)
        metrics.append(record)
    meta = metadata(cid, manifest, base)
    if cid is None:
        # Preserve distinct unidentified files instead of merging them into one run.
        meta['name'] = path.stem
        warnings.append(f'{path.name}: cell identity unavailable; no paired comparisons for this file.')
    elif cid not in manifest:
        meta['name'] = f'unknown_cell_{cid}'
        warnings.append(f'{path.name}: cell {cid} is absent from manifest.')
    for r in metrics + training:
        r.update(meta)
    return metrics, training, dict(meta, log_file=str(path), eval_headers=headers,
        error_lines=errors, bytes_at_scan=path.stat().st_size)


def read_evaluation_csvs(manifest, base, warnings):
    observations = []
    for cid in manifest:
        meta = metadata(cid, manifest, base)
        root = base/'results'/meta['name']
        for relative in ('dataframes/eval_metrics_in_progress.csv', 'dataframes/eval_metrics.csv', 'eval_metrics.csv'):
            path = root/relative
            if not path.exists():
                continue
            try:
                with path.open(newline='') as f:
                    source_rows = list(csv.DictReader(f))
                for idx, row in enumerate(source_rows, 2):
                    ep = integer(row.get('epoch'))
                    if ep is None:
                        continue
                    prefix = row.get('tag') or 'LSI_Diff'
                    if prefix == 'LSI_Diff' and meta['stage'] == 'frozen':
                        prefix = 'Frozen'
                    def add(config, metric_suffix=None, recon=False):
                        parsed = parse_config(config)
                        if not parsed:
                            return
                        record = dict(meta, epoch=ep, tag=prefix, config=config, description='', **parsed,
                            value_source='csv_full_precision', source_file=str(path), line=idx,
                            modified=path.stat().st_mtime)
                        for output, column in [('fid', 'fid'), ('kid', 'kid'), ('sw2', 'sw2'),
                            ('diversity', 'div'), ('lsi_gap', 'lsi_gap'), ('mse_eps', 'mse_gap_eps'), ('mse_score', 'mse_gap_score')]:
                            key = column+'_vae_recon' if recon else column+'_'+metric_suffix
                            record[output] = number(row.get(key))
                        if record['fid'] is not None:
                            observations.append(record)
                    if row.get('fid_vae_recon', '') != '':
                        add('VAE_Rec_eps@0', recon=True)
                    for col in row:
                        m = FID_COLUMN.match(col)
                        if not m or row[col] in ('', None):
                            continue
                        method, steps, suffix = m.groups()
                        method = {'rk4': 'rk4_ode', 'heun': 'heun_sde'}.get(method, method)
                        add(f'{method}@{steps}_{suffix}', col[len('fid_'):])
            except (OSError, csv.Error, ValueError, TypeError) as exc:
                warnings.append(f'{path}: could not finish reading this live CSV: {exc}')
    return observations


def metric_key(r):
    def norm(v):
        return round(v, 8) if finite(v) else None
    return (r['name'], r['epoch'], r['tag'], r['method'], r['steps'], norm(r['cfg']),
            norm(r['temperature']), r['init'], norm(r['horizon']))


def deduplicate(observations, warnings):
    groups = {}
    for r in observations:
        groups.setdefault(metric_key(r), []).append(r)
    merged = []
    for group in groups.values():
        chosen = max(group, key=lambda r: (r['value_source']=='csv_full_precision', r['modified'], r['line']))
        chosen = dict(chosen)
        vals = [r['fid'] for r in group if finite(r['fid'])]
        conflict = bool(vals and max(vals)-min(vals) > .011)
        chosen['conflicting_observations'] = conflict
        if conflict:
            warnings.append(f"{chosen['name']} ep {chosen['epoch']} {chosen['config']}: conflicting duplicate FIDs {min(vals):g}..{max(vals):g}; inspect all_observations.csv.")
        if not finite(chosen['fid']) or chosen['fid'] < 0:
            warnings.append(f"{chosen['name']} ep {chosen['epoch']} {chosen['config']}: invalid FID {chosen['fid']} retained in history.")
        merged.append(chosen)
    return sorted(merged, key=lambda r: (r['name'], r['epoch'], r['method'], str(r['cfg']), r['steps'], r['init']))


def primary(r, cfg=2.5):
    return (r['tag'] in PRODUCTION_TAGS and r['method']=='rk4' and r['steps']==20
        and close(r['cfg'], cfg) and close(r['temperature'], 1.) and r['init']=='gaussianT'
        and r['full_horizon'] is not None and close(r['horizon'], r['full_horizon'])
        and finite(r['fid']) and r['fid'] >= 0)


def make_latest(records, training, manifest, base, scans):
    metas = {metadata(c, manifest, base)['name']: metadata(c, manifest, base) for c in manifest}
    for r in records:
        metas.setdefault(r['name'], {k:r.get(k) for k in metadata(None, {}, base)})
    result = []
    for name, meta in metas.items():
        rr = [r for r in records if r['name']==name and r['tag'] in PRODUCTION_TAGS]
        tt = [r for r in training if r['name']==name]
        pp = [r for r in rr if primary(r)]
        ep = max((r['epoch'] for r in pp), default=None)
        top = next((r for r in pp if r['epoch']==ep), None)
        same = [r for r in rr if r['epoch']==ep] if ep is not None else []
        s = dict(meta, latest_printed_training_epoch=max((r['epoch'] for r in tt), default=None),
                 latest_any_eval_epoch=max((r['epoch'] for r in rr), default=None), primary_epoch=ep,
                 fid_cfg2_5=top['fid'] if top else None, fid_cfg1=None, fid_cfg3=None, recon_fid=None,
                 rk4_25_cfg2_5_fid=None, heun_20_cfg2_5_fid=None, value_source=top['value_source'] if top else '',
                 epoch_fraction=(f"{ep}/{meta['target_epoch']}" if ep is not None else ''),
                 primary_available=bool(top), final_epoch_observed=bool(ep is not None and ep==meta['target_epoch']))
        for r in same:
            if primary(r, 1.): s['fid_cfg1'] = r['fid']
            if primary(r, 3.): s['fid_cfg3'] = r['fid']
            if r['method']=='reconstruction': s['recon_fid'] = r['fid']
            if (close(r['cfg'], 2.5) and r['init']=='gaussianT' and close(r['horizon'], meta['full_horizon'])
                    and close(r['temperature'], 1.)):
                if r['method']=='rk4' and r['steps']==25:s['rk4_25_cfg2_5_fid']=r['fid']
                if r['method']=='heun' and r['steps']==20:s['heun_20_cfg2_5_fid']=r['fid']
        older = [r for r in tt if ep is not None and r['epoch']<=ep]
        if older:
            t = max(older, key=lambda r: (r['epoch'], r.get('modified', 0), r['line']))
            s.update(training_metrics_epoch=t['epoch'], posterior_var=t.get('post_var_mean'),
                     recon_mse=t.get('Rec'), terminal_factor_penalty=t.get('K_T'),
                     raw_score_mse=t.get('raw_score_mse'), head_score_mse=t.get('head_score_mse'))
        s['logged_errors'] = sum(len(x['error_lines']) for x in scans if x['name']==name)
        s['log_files'] = ';'.join(x['log_file'] for x in scans if x['name']==name)
        result.append(s)
    return sorted(result, key=lambda r: (r['cell_id']=='' or r['cell_id'] is None, integer(r['cell_id']) or 0, r['name']))


def paired_contrasts(records):
    by_cell = {}
    for r in records:
        if r['cell_id'] == '' or r['cell_id'] is None:
            continue
        by_cell.setdefault(int(r['cell_id']), []).append(r)
    comparisons = []
    # Exact design mappings only; callers suppress this for a different manifest.
    for seed in (0, 1):
        for a, b, label in [(2,0,'r0-minus-flat'),(4,0,'r05-minus-flat'),(6,0,'r1-minus-flat'),
                            (8,6,'r4-minus-r1'),(10,6,'decoupled-minus-r1'),(4,6,'r05-minus-r1')]:
            comparisons.append((a+seed, b+seed, label))
    for a in range(12,24,3):
        # Epoch 0 of each frozen branch is the unchanged source. Use it as the
        # source comparison at epoch 50; do not equate joint 800 with frozen 50.
        for b, label in [(a+1,'corrected-minus-naive-frozen'),(a+2,'dsm-minus-naive-frozen')]:
            comparisons.append((b, a, label))
    output = []
    for a, b, label in comparisons:
        for cfg in (1., 2.5, 3.):
            aa = {r['epoch']:r for r in by_cell.get(a, []) if primary(r,cfg)}
            bb = {r['epoch']:r for r in by_cell.get(b, []) if primary(r,cfg)}
            shared = set(aa) & set(bb)
            if not shared:
                continue
            ep = max(shared)
            u,v = aa[ep],bb[ep]
            if u['expected_eval_samples'] != v['expected_eval_samples']:
                continue
            output.append(dict(comparison=label, seed=u['seed'], epoch=ep, cfg=cfg, cell_a=a, cell_b=b,
                name_a=u['name'], name_b=v['name'], fid_a=u['fid'], fid_b=v['fid'],
                delta_fid=u['fid']-v['fid'], source_a=u['value_source'], source_b=v['value_source'],
                expected_eval_samples=u['expected_eval_samples']))
    return output


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row)) or ['no_rows']
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    temporary.replace(path)


def display(v):
    if v is None or v=='': return '—'
    if isinstance(v,float): return f'{v:.6g}'
    return str(v)


def make_report(latest, pairs, scans, warnings):
    visible = [r for r in latest if r['primary_available'] or r['latest_any_eval_epoch'] is not None or r['log_files']]
    lines = ['# Current CIFAR evaluations', '',
        'Snapshot UTC: '+datetime.now(timezone.utc).isoformat(timespec='seconds'), '',
        'Primary column: full-T Gaussian start, temperature 1, RK4-20, CFG 2.5. Epochs are shown explicitly. '
        'These are current completed measurements; a running job need not be finished.', '',
        'CSV values retain full precision. `log_rounded` FIDs have only the two decimal places printed by the trainer. '
        'Expected sample counts come from the manifest/run configuration, not an independent count in the Slurm table. '
        'An observed final epoch does not certify that the job finished successfully.', '',
        '| Cell | Run | Primary epoch | FID 2.5 | FID 1 | FID 3 | Recon FID | Training epoch | Source |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for r in visible:
        vals=[r['cell_id'],r['name'],r['primary_epoch'],r['fid_cfg2_5'],r['fid_cfg1'],r['fid_cfg3'],
              r['recon_fid'],r['latest_printed_training_epoch'],r['value_source']]
        lines.append('| '+' | '.join(display(v) for v in vals)+' |')
    if not visible: lines += ['', 'No production evaluation or training output found in the current snapshot.']
    lines += ['', 'All FIDs shown together in one row come from the same primary epoch. '
        'Missing CFG values stay blank. `latest_any_eval_epoch` in latest.csv can be newer if a sweep is only partly printed. '
        'Reconstruction and Gaussian-TK results remain distinct in eval_history.csv. Frozen epoch 0 is the source baseline.', '',
        '## Paired comparisons at the latest common epoch', '',
        'Negative delta favors the first arm. Different epochs are never subtracted.', '',
        '| Comparison | Seed | Epoch | CFG | FID A | FID B | A − B |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in pairs:
        lines.append('| '+' | '.join(display(r[k]) for k in ('comparison','seed','epoch','cfg','fid_a','fid_b','delta_fid'))+' |')
    lines += ['', '## Scan notes', '', f'{len(scans)} log files scanned. No checkpoint was loaded.']
    lines += ['- '+s for s in warnings]
    for scan in scans:
        lines += ['- '+scan['log_file']+': '+s for s in scan['error_lines']]
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir',type=Path,default=Path.cwd(),help='Extracted CIFAR sweep directory; default current directory')
    parser.add_argument('--logs',nargs='+',default=None,help='Log filenames or glob patterns; quote a pattern to expand it here')
    parser.add_argument('--logs-only',action='store_true',help='Ignore result CSVs even if available')
    parser.add_argument('--out-dir',type=Path,default=None,help='Default BASE/live_compiled; never modifies compiled/')
    parser.add_argument('--no-zip',action='store_true',help='Skip the small portable result ZIP')
    args = parser.parse_args(argv)
    base = args.base_dir.resolve(); out = args.out_dir.resolve() if args.out_dir else base/'live_compiled'
    warnings = []
    manifest = load_manifest(base,warnings)
    files = set()
    for pattern in args.logs or [str(base/'slurm_logs/*.out')]:
        p = Path(pattern)
        pattern = str(p if p.is_absolute() else base/p)
        matched = glob.glob(pattern)
        if not matched: warnings.append('No log files match '+pattern)
        files.update(Path(p).resolve() for p in matched if Path(p).is_file())
    observations = []; training = []; scans = []
    for path in sorted(files):
        try:
            ev,tr,scan = read_log(path,manifest,base,warnings)
            observations.extend(ev); training.extend(tr); scans.append(scan)
        except OSError as exc:
            warnings.append(str(path)+': '+str(exc))
    if not args.logs_only:
        observations.extend(read_evaluation_csvs(manifest,base,warnings))
    records = deduplicate(observations,warnings)
    latest = make_latest(records,training,manifest,base,scans)
    # Fixed contrasts require the delivered cell mapping; custom manifests still get histories.
    expected = fallback_manifest()
    same_design = all(str(manifest.get(i,{}).get('result_name'))==r['result_name'] for i,r in expected.items())
    pairs = paired_contrasts(records) if same_design else []
    if not same_design:warnings.append('Manifest differs from the delivered 24-cell mapping; fixed paired contrasts disabled.')
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'latest.csv',latest)
    write_csv(out/'eval_history.csv',records)
    write_csv(out/'all_observations.csv',observations)
    write_csv(out/'training_history.csv',training)
    write_csv(out/'paired_latest_common_epoch.csv',pairs,
        fields=['comparison','seed','epoch','cfg','cell_a','cell_b','name_a','name_b','fid_a','fid_b','delta_fid','source_a','source_b','expected_eval_samples'])
    report = make_report(latest,pairs,scans,list(dict.fromkeys(warnings)))
    (out/'REPORT.md').write_text(report)
    (out/'scan_notes.json').write_text(json.dumps(dict(logs=scans,warnings=warnings,logs_only=args.logs_only),indent=2))
    print(report)
    print('Wrote:',out)
    if not args.no_zip:
        target = out/'live_cifar_snapshot.zip'
        with zipfile.ZipFile(target.with_suffix('.zip.tmp'),'w',zipfile.ZIP_DEFLATED) as z:
            for name in ('REPORT.md','latest.csv','eval_history.csv','all_observations.csv','training_history.csv','paired_latest_common_epoch.csv','scan_notes.json'):
                z.write(out/name,name)
        target.with_suffix('.zip.tmp').replace(target)
        print('Portable snapshot:',target)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
