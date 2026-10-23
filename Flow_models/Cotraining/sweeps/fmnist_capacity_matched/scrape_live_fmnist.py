#!/usr/bin/env python3
"""Scrape unfinished capacity-matched FMNIST CSEM jobs; standard library only.

SAVE THIS SCRIPT OUTSIDE THE EXPERIMENT FOLDER. That bundle hashes root *.py
files, so putting a new script in its root can invalidate queued/resumed jobs.
From the experiment folder, with this script in its parent:
    python ../scrape_live_fmnist.py --base-dir .
    python ../scrape_live_fmnist.py --base-dir . --logs-only

Reads bounded snapshots of [CONFIG]/[EVAL]/[TRAIN] JSON in Slurm logs and, when
available, results/<run>/{config.json,eval.csv,loss.csv}. No training imports,
Torch, checkpoints, scheduler commands, or third-party packages are used.
Writes BASE/live_compiled and live_fmnist_snapshot.zip. Safe before any job ends.
"""
from __future__ import annotations
import argparse
import codecs
import csv
from datetime import datetime, timezone
import glob
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
import zipfile

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
TAG = re.compile(r'\[(CONFIG|EVAL|TRAIN|DIAGNOSTICS|BIAS_WITNESS)\]\s*(\{.*)$')
CELL = re.compile(r'(?:fmcap_c|cell[_-]?)(\d+)(?:_|\.|$)')
ERRORS = ('traceback (most recent call last)', 'out of memory', 'slurmstepd: error:',
          'source/recipe changed', 'resume configuration/source mismatch', 'insufficient balance')
SEEDS = (42,43,44,45,46)
ARMS = (('flat',2,'decoupled',1,'reference_two'),('stack2',2,'full',2,'reference_two'),
        ('flat_three_encoder',3,'decoupled',1,'budget_three'),('stack3',3,'full',3,'budget_three'))
PAIRS = (('stack2-minus-flat','stack2','flat','primary: exact architecture and capacity'),
         ('stack3-minus-flat-three-encoder','stack3','flat_three_encoder','primary: exact architecture and capacity'),
         ('stack3-minus-flat','stack3','flat','secondary: encoder allocation differs'),
         ('stack3-minus-stack2','stack3','stack2','secondary: encoder allocation differs'))
COMMON = dict(epochs=500,eval_samples=10000,t_min=2e-5,t_max=2.,decode_time=1e-4,
    rk4_steps=20,joint_width=192,joint_depth=18,joint_heads=6,batch=128,
    rep_multiplier=.5,score_weight=.6,terminal_weight=1.,band_mode='fixed',clip_mode='global',
    lr_vae=5e-4,lr_head=2e-4,perc_weight=.85,gan_weight=.0025,ema=.9997,
    disc_start=25,lr_disc=1e-4,label_dropout=.1,tmin_band_weight='auto')
SCIENCE = tuple(COMMON)+('seed','blocks','topology','stochastic_blocks','capacity_family','arm',
                         'z1_channels','z2_channels','z3_channels','max_batches')
META = ('cell_id','name','arm','seed','stochastic_blocks','storage_factors','topology','r','stage','target_epoch')
PROTOCOL = ('cfg','method','steps','init','num_eval','backend','horizon','t_min','decode_time','temperature')
EVAL_FIELDS = META+('epoch',)+PROTOCOL+('fid_generated','fid_posterior_recon','value_source','source_file',
    'line','config_valid','config_basis','active','conflicting_observations','duplicate_count')
PAIR_FIELDS = ('comparison','role','seed','epoch')+PROTOCOL+('cell_a','cell_b','name_a','name_b',
    'fid_a','fid_b','a_minus_b','recon_a','recon_b','recon_a_minus_b','source_a','source_b')
# Two-sided Student t(.975), degrees of freedom 1..4: this study has <=5 seeds.
T95 = {1:12.706204736432095,2:4.302652729696142,3:3.182446305284263,4:2.7764451051977987}


def number(value):
    try:return float(value) if value is not None and str(value).strip() else None
    except (ValueError,TypeError):return None


def integer(value):
    x=number(value)
    return int(x) if x is not None and math.isfinite(x) and x.is_integer() else None


def finite(value):
    return value is not None and math.isfinite(value)


def close(a,b):
    return finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)


def snapshot_lines(path):
    """Read bytes present at open time; ignore an unfinished final line."""
    with Path(path).open('rb') as source:
        remaining=os.fstat(source.fileno()).st_size
        decoder=codecs.getincrementaldecoder('utf-8')(errors='replace');carry='';line_no=0
        while remaining:
            chunk=source.read(min(262144,remaining))
            if not chunk:break
            remaining-=len(chunk)
            parts=re.split(r'[\r\n]',carry+decoder.decode(chunk));carry=parts.pop()
            for line in parts:
                line_no+=1
                yield line_no,ANSI.sub('',line).strip()


def csv_snapshot(path):
    # Atomic CSV replacement by the trainer is safe: the open descriptor keeps
    # its original bounded snapshot, even if the path is replaced concurrently.
    with Path(path).open('rb') as source:
        content=source.read(os.fstat(source.fileno()).st_size)
    if content and not content.endswith((b'\n',b'\r')):
        end=max(content.rfind(b'\n'),content.rfind(b'\r'));content=content[:end+1]
    reader=csv.DictReader(io.StringIO(content.decode('utf-8',errors='replace')))
    return [(line,row) for line,row in enumerate(reader,2) if None not in row and None not in row.values()]


def fallback_manifest():
    result={}
    for seed in SEEDS:
        for arm,blocks,topology,depth,family in ARMS:
            cid=len(result)
            cfg=dict(COMMON,seed=seed,blocks=blocks,topology=topology,
                     stochastic_blocks=depth,capacity_family=family,arm=arm,z1_channels=4,
                     z2_channels=3 if blocks==2 else 1,z3_channels=0 if blocks==2 else 2,max_batches=0)
            result[cid]=dict(cell_id=cid,name=f'{arm}_s{seed}',stage='joint',config=cfg)
    return result


def load_manifest(base,warnings):
    path=base/'manifest.json'
    if path.exists():
        raw=json.loads(path.read_text())
        if not isinstance(raw,list):raise ValueError('manifest.json must contain a list of cells')
        result={int(r['cell_id']):r for r in raw}
        if len(result)!=len(raw):raise ValueError('Duplicate manifest cell IDs')
    elif (base/'manifest.csv').exists():
        result={};expected=fallback_manifest()
        for _,row in csv_snapshot(base/'manifest.csv'):
            cid=int(row['cell_id']);known=expected.get(cid,{})
            cfg=dict(known.get('config',{})) if known.get('name')==row.get('name') else {}
            for key,value in row.items():
                if key not in ('cell_id','name','stage') and value!='':
                    cfg[key]=number(value) if number(value) is not None else value
            result[cid]=dict(cell_id=cid,name=row['name'],stage=row.get('stage','joint'),config=cfg)
        warnings.append('Using CSV manifest; absent protocol fields use defaults only for recognized matched-study cells.')
    else:
        result=fallback_manifest()
        warnings.append('No manifest found: assuming the delivered 20-cell capacity-matched FMNIST study. CONFIG records are checked against that design.')
    if len({r['name'] for r in result.values()})!=len(result):raise ValueError('Duplicate manifest run names')
    for cell in result.values():
        if Path(cell['name']).name!=cell['name']:raise ValueError('Run names must be single directory names')
    return result


def meta_for(cid,manifest,actual=None,basis='manifest'):
    row=manifest.get(cid,{})
    expected=row.get('config',{});actual=actual or {}
    cfg=dict(expected);cfg.update(actual)
    mismatch=[key for key in SCIENCE if key in expected and key in actual and str(expected[key])!=str(actual[key])
              and not (number(expected[key]) is not None and close(number(expected[key]),number(actual[key])))]
    known=bool(row)
    return dict(cell_id=cid if cid is not None else '',name=row.get('name',actual.get('name','')),
        arm=cfg.get('arm',''),seed=integer(cfg.get('seed')),stochastic_blocks=integer(cfg.get('stochastic_blocks')),
        storage_factors=integer(cfg.get('blocks')),topology=cfg.get('topology',''),r=number(cfg.get('rep_multiplier')),
        stage=row.get('stage',cfg.get('stage','unknown')),target_epoch=integer(cfg.get('epochs')),
        horizon=number(cfg.get('t_max')),t_min=number(cfg.get('t_min')),decode_time=number(cfg.get('decode_time')),
        temperature=1.,config_valid=known and not mismatch,config_basis=basis,
        config_mismatch=','.join(mismatch),expected_eval_samples=integer(cfg.get('eval_samples')),
        recipe_key=json.dumps({key:(number(cfg[key]) if number(cfg.get(key)) is not None else cfg.get(key))
                               for key in COMMON},sort_keys=True))


def normalize_eval(raw,meta,path,line,source,warnings):
    row=dict(meta,epoch=integer(raw.get('epoch')),cfg=number(raw.get('cfg')),method=raw.get('method','rk4'),steps=integer(raw.get('steps')),
        init=raw.get('init','unknown'),num_eval=integer(raw.get('num_eval')),backend=raw.get('backend','unknown'),
        fid_generated=number(raw.get('fid_generated')),fid_posterior_recon=number(raw.get('fid_posterior_recon')),
        value_source=source,source_file=str(path),line=line,modified=path.stat().st_mtime,
        active=True,conflicting_observations=False,duplicate_count=1)
    if row['epoch'] is None or row['cfg'] is None or row['steps'] is None or row['num_eval'] is None:
        warnings.append(f'{path.name}:{line}: incomplete evaluation fields; skipped');return None
    if not finite(row['fid_generated']) or row['fid_generated']<0:
        warnings.append(f'{path.name}:{line}: invalid generated FID retained in history, excluded from comparisons')
    if row['fid_posterior_recon'] is not None and (not finite(row['fid_posterior_recon']) or row['fid_posterior_recon']<0):
        warnings.append(f'{path.name}:{line}: invalid reconstruction FID retained in history, excluded from reconstruction deltas')
    return row


def read_log(path,manifest,warnings):
    match=CELL.search(path.name);cid=int(match.group(1)) if match else None
    meta=meta_for(cid,manifest);events=[];training=[];diagnostics=[];errors=[];identities=set();malformed=0
    by_name={r['name']:i for i,r in manifest.items()}
    def remember():
        if meta['name']:identities.add(meta['name'])
    remember()
    for line_no,line in snapshot_lines(path):
        if any(s in line.lower() for s in ERRORS):errors.append(dict(line=line_no,message=line[:400]))
        found=TAG.search(line)
        if not found:continue
        tag,text=found.groups()
        try:data=json.loads(text)
        except (ValueError,TypeError):malformed+=1;continue
        if not isinstance(data,dict):malformed+=1;continue
        if tag=='CONFIG':
            header=integer(data.get('cell_id'));name=data.get('name')
            if header is None and name in by_name:header=by_name[name]
            if header is not None:
                if cid is not None and header!=cid:
                    warnings.append(f'{path.name}:{line_no}: CONFIG cell {header} differs from filename/previous cell {cid}; subsequent rows use CONFIG')
                cid=header
            meta=meta_for(cid,manifest,data,'log_CONFIG')
            if name and meta['name']!=name:
                meta['config_valid']=False;meta['config_mismatch']+=' name/cell identity'
            if not meta['name']:meta['name']=path.stem
            if not meta['config_valid']:warnings.append(f'{path.name}:{line_no}: unrecognized/mismatched CONFIG ({meta["config_mismatch"]}); rows excluded from primary comparisons')
            remember();continue
        if not meta['name']:meta['name']=path.stem;remember()
        if tag=='EVAL':
            row=normalize_eval(data,meta,path,line_no,'log_json_full_precision',warnings)
            if row:events.append(row)
        elif tag=='TRAIN':
            if integer(data.get('epoch')) is not None:
                training.append(dict(data,**{k:meta.get(k) for k in META},epoch=integer(data['epoch']),
                    value_source='log_json_full_precision',source_file=str(path),line=line_no,
                    modified=path.stat().st_mtime,config_valid=meta['config_valid'],active=True))
        else:
            values=data.get('summary',[]) if tag=='DIAGNOSTICS' else data.get('results',[])
            for value in values:
                if not isinstance(value,dict):continue
                record=dict(value,**{k:meta.get(k) for k in META},epoch=integer(data.get('epoch')),
                    section=tag,source_file=str(path),line=line_no,config_valid=meta['config_valid'])
                if tag=='BIAS_WITNESS':record.update(metric='witness/bias_squared_lower',mean=value.get('lower'))
                diagnostics.append(record)
    scan=dict(file=str(path),bytes_at_scan=path.stat().st_size,names=sorted(identities),
              evaluation_rows=len(events),training_rows=len(training),malformed_json_lines=malformed,error_lines=errors)
    if malformed:warnings.append(f'{path.name}: {malformed} malformed completed JSON lines skipped')
    if events and not any(r['config_valid'] for r in events):warnings.append(f'{path.name}: no identified valid experiment rows; retained for inspection only')
    return events,training,diagnostics,scan


def read_result_csvs(base,manifest,warnings):
    events=[];training=[];canonical_eval={};canonical_train={};invalid_names=set();configs={}
    for cid,cell in manifest.items():
        name=cell['name'];root=base/'results'/name;actual={};basis='manifest'
        path=root/'config.json'
        if path.exists():
            try:
                actual=json.loads(path.read_text());basis='run_config'
                if not isinstance(actual,dict):raise ValueError('configuration is not an object')
                configs[name]=actual
            except (OSError,ValueError) as exc:
                warnings.append(f'{path}: cannot read configuration: {exc}');invalid_names.add(name)
        meta=meta_for(cid,manifest,actual,basis)
        if not meta['config_valid']:
            invalid_names.add(name);warnings.append(f'{name}: configuration mismatch ({meta["config_mismatch"]}); excluded from comparisons')
        for filename,target,canonical in [('eval.csv',events,canonical_eval),('loss.csv',training,canonical_train)]:
            path=root/filename
            if not path.exists():continue
            try:
                snapshot=csv_snapshot(path)
                parsed=[]
                for line,raw in snapshot:
                    if filename=='eval.csv':row=normalize_eval(raw,meta,path,line,'csv_full_precision',warnings)
                    else:
                        row=None
                        if integer(raw.get('epoch')) is not None:
                            row={k:(number(v) if number(v) is not None else v) for k,v in raw.items()}
                            row.update({k:meta.get(k) for k in META})
                            row.update(epoch=integer(raw['epoch']),value_source='csv_full_precision',source_file=str(path),
                                line=line,modified=path.stat().st_mtime,config_valid=meta['config_valid'],active=True)
                    if row:parsed.append(row)
                target.extend(parsed)
                # Existing empty canonical CSV suppresses old rolled-back log
                # epochs. A concurrently newer log row appears on the next scan.
                canonical[name]={eval_key(r) if filename=='eval.csv' else (r['name'],r['epoch']) for r in parsed}
            except (OSError,csv.Error,ValueError,TypeError) as exc:
                warnings.append(f'{path}: could not finish this CSV snapshot: {exc}')
    return events,training,canonical_eval,canonical_train,invalid_names,configs


def eval_key(row):
    return (row['name'],row['epoch'])+tuple(row.get(k) for k in PROTOCOL)


def merge_rows(rows,warnings,canonical=None,invalid_names=None,training=False):
    groups={};canonical=canonical or {};invalid_names=invalid_names or set()
    for row in rows:
        key=(row['name'],row['epoch']) if training else eval_key(row)
        active=row['name'] not in invalid_names and row.get('config_valid',False)
        if row['name'] in canonical and key not in canonical[row['name']]:active=False
        row['active']=active
        groups.setdefault(key,[]).append(row)
    result=[]
    for key,values in groups.items():
        selected=dict(max(values,key=lambda r:(r['value_source']=='csv_full_precision',r['modified'],r['line'])))
        if not training:
            observed=[r['fid_generated'] for r in values if finite(r['fid_generated'])]
            recon=[r['fid_posterior_recon'] for r in values if finite(r['fid_posterior_recon'])]
            conflicting=bool(observed and not close(min(observed),max(observed))) or bool(recon and not close(min(recon),max(recon)))
            selected.update(conflicting_observations=conflicting,duplicate_count=len(values))
            if conflicting:
                warnings.append(f'{selected["name"]} epoch {selected["epoch"]} CFG {selected["cfg"]}: conflicting duplicate FIDs; inspect all_observations.csv. CSV takes precedence when available.')
            # Conflicting log-only attempts cannot be selected scientifically.
            if conflicting and selected['value_source']!='csv_full_precision':selected['active']=False
        result.append(selected)
    return sorted(result,key=lambda r:(str(r['name']),r['epoch'],str(r.get('cfg','')),str(r.get('init','')),r.get('num_eval',0)))


def primary(row,cfg=3.):
    return (row.get('active',False) and row.get('stage')=='joint' and close(row.get('cfg'),cfg)
        and row.get('method')=='rk4' and row.get('steps')==20 and row.get('init')=='gaussian' and row.get('num_eval')==10000
        and row.get('backend')=='inception-legacy' and close(row.get('horizon'),2.)
        and close(row.get('t_min'),2e-5) and close(row.get('decode_time'),1e-4)
        and close(row.get('temperature'),1.) and finite(row.get('fid_generated')) and row['fid_generated']>=0)


def paired_history(records,warnings):
    indices={}
    for row in records:
        if not (primary(row,1.) or primary(row,3.)):continue
        if row.get('arm') not in {a[0] for a in ARMS} or row.get('seed') is None:continue
        key=(row['epoch'],)+tuple(row.get(k) for k in PROTOCOL)+(row.get('recipe_key'),)
        group=indices.setdefault((row['arm'],row['seed']),{})
        if key in group:
            warnings.append(f'Duplicate arm/seed/protocol for {row["arm"]}, seed {row["seed"]}; ambiguous pairing skipped')
            group[key]=None
        else:group[key]=row
    result=[]
    seeds=sorted({seed for _,seed in indices})
    for label,arm_a,arm_b,role in PAIRS:
        for seed in seeds:
            aa=indices.get((arm_a,seed),{});bb=indices.get((arm_b,seed),{})
            for key in sorted(aa.keys() & bb.keys()):
                a,b=aa[key],bb[key]
                if a is None or b is None:continue
                ra,rb=a['fid_posterior_recon'],b['fid_posterior_recon']
                result.append(dict(comparison=label,role=role,seed=seed,epoch=a['epoch'],
                    **{k:a[k] for k in PROTOCOL},cell_a=a['cell_id'],cell_b=b['cell_id'],name_a=a['name'],name_b=b['name'],
                    fid_a=a['fid_generated'],fid_b=b['fid_generated'],a_minus_b=a['fid_generated']-b['fid_generated'],
                    recon_a=ra,recon_b=rb,recon_a_minus_b=ra-rb if finite(ra) and finite(rb) and ra>=0 and rb>=0 else None,
                    source_a=a['value_source'],source_b=b['value_source']))
    return result


def latest_pairs(history):
    result={}
    for row in history:
        key=(row['comparison'],row['seed'])+tuple(row[k] for k in PROTOCOL)
        if key not in result or row['epoch']>result[key]['epoch']:result[key]=row
    return list(result.values())


def moments(values):
    result=dict(n=len(values),mean=statistics.mean(values) if values else None,sd=None,se=None,ci95_low=None,ci95_high=None)
    if len(values)>=2:
        sd=statistics.stdev(values);se=sd/math.sqrt(len(values));critical=T95.get(len(values)-1)
        result.update(sd=sd,se=se)
        if critical is not None:result.update(ci95_low=result['mean']-critical*se,ci95_high=result['mean']+critical*se)
    return result


def seed_summaries(history,manifest,endpoint=False):
    all_seeds=sorted({integer(r['config'].get('seed')) for r in manifest.values()}-{None})
    cohorts=(('all_seeds',all_seeds),('new_confirmation',[s for s in (44,45,46) if s in all_seeds]))
    result=[]
    for cohort,expected in cohorts:
        if not expected:continue
        for label,_,_,role in PAIRS:
            for cfg in (1.,3.):
                by_seed={seed:{r['epoch']:r for r in history if r['comparison']==label and r['seed']==seed and r['cfg']==cfg} for seed in expected}
                available={s:d for s,d in by_seed.items() if d}
                shared=set.intersection(*(set(d) for d in available.values())) if available else set()
                epoch=500 if endpoint else max(shared,default=None)
                values=[d[epoch] for d in by_seed.values() if epoch in d] if epoch is not None else []
                for metric in ('a_minus_b','recon_a_minus_b'):
                    good=[r for r in values if finite(r[metric])]
                    result.append(dict(cohort=cohort,comparison=label,role=role,cfg=cfg,epoch=epoch,metric=metric,
                        expected_pairs=len(expected),available_pairs=len(good),complete=len(good)==len(expected),
                        missing_seeds=','.join(str(s) for s in expected if not any(r['seed']==s for r in good)),
                        note='No common epoch across available seeds' if epoch is None and available else '',
                        **moments([r[metric] for r in good])))
    return result


def make_latest(records,training,manifest,scans,base):
    result=[]
    for cid,cell in manifest.items():
        meta=meta_for(cid,manifest);name=cell['name']
        ev=[r for r in records if r['name']==name and r.get('active')]
        tr=[r for r in training if r['name']==name and r.get('active')]
        candidates=[r for r in ev if primary(r)];top=max(candidates,key=lambda r:r['epoch'],default=None)
        epoch=top['epoch'] if top else None
        one=next((r for r in ev if r['epoch']==epoch and primary(r,1.)),None)
        latest_train=max(tr,key=lambda r:r['epoch'],default=None)
        associated=[s for s in scans if name in s['names']]
        row={k:meta[k] for k in META}
        row.update(primary_epoch=epoch,fid_cfg3=top['fid_generated'] if top else None,
            fid_cfg1=one['fid_generated'] if one else None,
            recon_fid=top['fid_posterior_recon'] if top and finite(top['fid_posterior_recon']) and top['fid_posterior_recon']>=0 else None,
            latest_any_eval_epoch=max((r['epoch'] for r in ev),default=None),
            training_epoch=latest_train['epoch'] if latest_train else None,
            value_source=top['value_source'] if top else '',primary_available=bool(top),
            final_epoch_observed=bool(top and epoch==meta['target_epoch']),
            completion_marker_present=(base/'results'/name/'complete.json').exists(),
            logged_error_count=sum(len(s['error_lines']) for s in associated),
            log_files=';'.join(s['file'] for s in associated))
        if latest_train:
            for key in ('seconds','total','recon','score','terminal','band','lpips','gan_g','vae_grad_norm','head_grad_norm',
                        'joint_clip_factor','score_1','score_2','score_3','var_1','var_2','var_3'):
                row['train_'+key]=latest_train.get(key)
        result.append(row)
    return result


def display(value):
    if value is None or value=='':return '—'
    if isinstance(value,float):return f'{value:.6g}'
    return str(value)


def make_report(latest,pairs,summaries,warnings,scans,logs_only):
    lines=['# Current FMNIST evaluations','','Snapshot UTC: '+datetime.now(timezone.utc).isoformat(timespec='seconds'),'',
        'Primary: full-T Gaussian start (T=2), temperature 1, RK4-20, CFG 3, 10,000 samples, legacy Inception. '
        'Only completed evaluation records are shown; jobs may still be running. CSV and JSON log values retain full precision.','',
        'Flat has one stochastic block despite its internal storage groups. All FIDs in a row use the same primary epoch; '
        'CFG 1 stays blank if its matching measurement is unavailable. Latest-any-eval can be newer during a partly printed sweep.','',
        '| Cell | Run | Blocks | Primary epoch | FID 3 | FID 1 | Recon FID | Training epoch | Source |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for row in latest:
        lines.append('| '+' | '.join(display(row.get(k)) for k in ('cell_id','name','stochastic_blocks','primary_epoch','fid_cfg3','fid_cfg1','recon_fid','training_epoch','value_source'))+' |')
    lines+=['','Reconstruction is the same-epoch posterior reconstruction FID. Oracle starts, smaller sampling checks, '
        'RK4-50, and synthetic features stay separate in eval_history.csv. A final epoch or completion marker does not validate job success.','',
        '## Paired comparisons at the latest common epoch','','A is the first named arm, B the second. Delta = A − B; negative favors A. '
        'Epoch, CFG, solver, start, sample count, backend, horizon, and decode settings must match.','',
        '| Comparison | Seed | Epoch | CFG | FID A | FID B | A − B | Recon A − B |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in pairs:
        lines.append('| '+' | '.join(display(row[k]) for k in ('comparison','seed','epoch','cfg','fid_a','fid_b','a_minus_b','recon_a_minus_b'))+' |')
    lines+=['','## Paired seed summaries at a common epoch','','All included seeds use one common epoch within each summary. '
        'Missing seeds are explicit. Seeds 44–46 provide new confirmation; 42/43 were used historically to choose r=0.5. '
        'Intervals describe paired training-seed variation, are pointwise, and do not measure FID-estimator uncertainty.','',
        '| Cohort | Comparison | Epoch | n / expected | Mean A − B (CFG 3) | 95% interval |',
        '|---|---|---:|---:|---:|---|']
    for row in summaries:
        if row['cfg']!=3. or row['metric']!='a_minus_b' or not row['n']:continue
        ci='—' if row['ci95_low'] is None else f'[{row["ci95_low"]:+.5f}, {row["ci95_high"]:+.5f}]'
        lines.append(f'| {row["cohort"]} | {row["comparison"]} | {row["epoch"]} | {row["n"]} / {row["expected_pairs"]} | {row["mean"]:+.6g} | {ci} |')
    lines+=['','The prespecified endpoint remains epoch 500 / CFG 3; fixed-endpoint summaries are in endpoint_pairs.csv. '
        'Historical FID 4.3040 is a reproduction target and is never inserted as a current measurement.','',
        '## Scan notes','',f'{len(scans)} logs scanned. Mode: '+('logs only.' if logs_only else 'logs plus available result CSVs.'),
        'No training module or checkpoint was loaded. Existing CSV snapshots take precedence over old log attempts; '
        'log-only observations absent from a readable current CSV are retained as inactive history.']
    lines+=['- '+message for message in dict.fromkeys(warnings)]
    for scan in scans:
        for error in scan['error_lines']:lines.append(f'- {scan["file"]}:{error["line"]}: {error["message"]}')
    return '\n'.join(lines)+'\n'


def write_csv(path,rows,fields=None):
    rows=list(rows);fields=fields or list(dict.fromkeys(k for row in rows for k in row)) or ['no_rows']
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir',type=Path,default=Path.cwd())
    parser.add_argument('--logs',nargs='+',help='Log files or quoted globs; relative paths resolve under BASE')
    parser.add_argument('--logs-only',action='store_true',help='Ignore result CSVs and use JSON log records')
    parser.add_argument('--out-dir',type=Path,help='Default BASE/live_compiled')
    parser.add_argument('--no-zip',action='store_true')
    parser.add_argument('--self-test',action='store_true',help='Run independent synthetic parser/comparison checks')
    args=parser.parse_args(argv)
    if args.self_test:return self_test()
    base=args.base_dir.resolve();out=args.out_dir.resolve() if args.out_dir else base/'live_compiled'
    if not base.is_dir():parser.error(f'Experiment directory does not exist: {base}')
    if Path(__file__).resolve().parent==base:
        parser.error('Move scrape_live_fmnist.py OUTSIDE the experiment directory: the bundle hashes root Python files. Then run python ../scrape_live_fmnist.py --base-dir .')
    if out==base or any(out==base/d or (base/d) in out.parents for d in ('results','slurm_logs','status','pinned','provenance')):
        parser.error('Output must be separate from experiment inputs (default: BASE/live_compiled)')
    warnings=[]
    try:manifest=load_manifest(base,warnings)
    except (OSError,ValueError,KeyError,csv.Error) as exc:parser.error(f'Cannot read experiment manifest: {exc}')
    files=set()
    for pattern in args.logs or ['slurm_logs/*.out','slurm_logs/*.err']:
        path=Path(pattern);pattern=str(path if path.is_absolute() else base/path)
        matches=glob.glob(pattern)
        if not matches and args.logs:warnings.append('No logs match '+pattern)
        files.update(Path(p).resolve() for p in matches if Path(p).is_file())
    if not files:warnings.append('No Slurm logs found; available result CSVs will still be read unless --logs-only.')
    observations=[];training=[];diagnostics=[];scans=[]
    for path in sorted(files):
        try:
            ev,tr,di,scan=read_log(path,manifest,warnings)
            observations+=ev;training+=tr;diagnostics+=di;scans.append(scan)
        except OSError as exc:warnings.append(f'{path}: {exc}')
    canonical_eval={};canonical_train={};invalid=set()
    if not args.logs_only:
        ev,tr,canonical_eval,canonical_train,invalid,_=read_result_csvs(base,manifest,warnings)
        observations+=ev;training+=tr
    records=merge_rows(observations,warnings,canonical_eval,invalid)
    train=merge_rows(training,warnings,canonical_train,invalid,training=True)
    history=paired_history(records,warnings);pairs=latest_pairs(history)
    summaries=seed_summaries(history,manifest);endpoints=seed_summaries(history,manifest,endpoint=True)
    latest=make_latest(records,train,manifest,scans,base)
    report=make_report(latest,pairs,summaries,warnings,scans,args.logs_only)
    out.mkdir(parents=True,exist_ok=True)
    outputs=[('latest.csv',latest,None),('eval_history.csv',records,EVAL_FIELDS),
        ('all_observations.csv',observations,EVAL_FIELDS),('training_history.csv',train,None),
        ('diagnostics.csv',diagnostics,None),('paired_history.csv',history,PAIR_FIELDS),
        ('paired_latest_common_epoch.csv',pairs,PAIR_FIELDS),('paired_seed_summary.csv',summaries,None),
        ('endpoint_pairs.csv',endpoints,None)]
    for name,values,fields in outputs:write_csv(out/name,values,fields)
    (out/'REPORT.md').write_text(report)
    (out/'scan_notes.json').write_text(json.dumps(dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),base_dir=str(base),
        logs_only=args.logs_only,logs=scans,warnings=list(dict.fromkeys(warnings)),manifest_cells=len(manifest),
        evaluation_rows=len(records),active_evaluation_rows=sum(r['active'] for r in records)),indent=2)+'\n')
    if not args.no_zip:
        target=out/'live_fmnist_snapshot.zip';temp=target.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
            for name in [item[0] for item in outputs]+['REPORT.md','scan_notes.json']:
                archive.write(out/name,name)
        temp.replace(target)
    print(report);print('Wrote:',out)
    if not args.no_zip:print('Portable snapshot:',out/'live_fmnist_snapshot.zip')
    return 0


# self_test is defined below so the file remains a single, portable deliverable.
def self_test():
    """Independent fixtures; no dataset, trainer, Torch, or real jobs required."""
    import contextlib
    checks=[]
    with tempfile.TemporaryDirectory(prefix='scrape-fmnist-') as temp:
        base=Path(temp);(base/'slurm_logs').mkdir();manifest=fallback_manifest()
        (base/'manifest.json').write_text(json.dumps(list(manifest.values())))
        def evaluation(epoch=50,cfg=3.,fid=4.3040228295,**other):
            row=dict(epoch=epoch,cfg=cfg,steps=20,init='gaussian',num_eval=10000,
                     backend='inception-legacy',fid_generated=fid,fid_posterior_recon=2.738)
            row.update(other);return row
        def logfile(cid,values,train=None,partial=False):
            cell=manifest[cid];path=base/'slurm_logs'/f'fmcap_c{cid}_1090160.out'
            cfg=dict(cell['config'],name=cell['name'],cell_id=cid)
            lines=['[CONFIG] '+json.dumps(cfg)]+['\x1b[32m[EVAL] '+json.dumps(v)+'\x1b[0m' for v in values]
            if train:lines+=['[TRAIN] '+json.dumps(train)]
            path.write_text('\n'.join(lines)+'\n'+('[EVAL] {"epoch": 999' if partial else ''))
            return path
        path=logfile(1,[evaluation(),evaluation(cfg=1.,fid=5.3848377148),
            evaluation(fid=.01,num_eval=1024),evaluation(fid=.02,init='oracle_qT'),
            evaluation(fid=.03,steps=50),evaluation(fid=.04,backend='synthetic-smoke-only')],
            train=dict(epoch=63,seconds=29.3,recon=.021,score=.02),partial=True)
        warnings=[];ev,tr,_,scan=read_log(path,manifest,warnings)
        assert len(ev)==6 and len(tr)==1 and ev[0]['fid_generated']==4.3040228295
        records=merge_rows(ev,warnings)
        assert sum(primary(r) for r in records)==1
        heun=dict(records[0],method='heun')
        assert not primary(heun)
        assert tr[0]['epoch']==63 and scan['malformed_json_lines']==0
        checks+=['Full-precision JSON, ANSI sequences, bounded partial lines, train rows, sample-count/solver/start/backend separation']
        unknown=base/'slurm_logs'/'slurm_1090999.out';unknown.write_text(path.read_text())
        assert read_log(unknown,manifest,[])[0][0]['cell_id']==1
        checks+=['CONFIG identity works without a cell-number filename']
        path0=logfile(0,[evaluation(fid=5.),evaluation(epoch=100,cfg=1.,fid=6.)])
        path1=logfile(1,[evaluation(fid=4.3),evaluation(epoch=100,cfg=1.,fid=5.),evaluation(epoch=100,fid=4.1)])
        left=read_log(path0,manifest,[])[0];right=read_log(path1,manifest,[])[0]
        root=base/'results'/'stack2_s42';root.mkdir(parents=True)
        (root/'config.json').write_text(json.dumps(manifest[1]['config']))
        def put_csv(rows):
            with (root/'eval.csv').open('w',newline='') as target:
                writer=csv.DictWriter(target,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        put_csv([evaluation(fid=4.31),evaluation(epoch=100,cfg=1.,fid=5.)])
        warnings=[];cev,ctr,canonical,ct,invalid,_=read_result_csvs(base,manifest,warnings)
        records=merge_rows(left+right+cev,warnings,canonical,invalid)
        primary_stack=[r for r in records if r['name']=='stack2_s42' and primary(r)]
        assert len(primary_stack)==1 and primary_stack[0]['epoch']==50 and primary_stack[0]['fid_generated']==4.31
        assert primary_stack[0]['conflicting_observations']
        assert not next(r for r in records if r['name']=='stack2_s42' and r['epoch']==100 and r['cfg']==3.)['active']
        pairs=latest_pairs(paired_history(records,warnings))
        p3=next(r for r in pairs if r['cfg']==3.);p1=next(r for r in pairs if r['cfg']==1.)
        assert p3['epoch']==50 and close(p3['a_minus_b'],-.69) and p1['epoch']==100
        unmatched=[r for r in left if r['epoch']==50]+[r for r in right if r['epoch']==100]
        assert not paired_history(merge_rows(unmatched,[]),[])
        other_recipe=[dict(r,recipe_key='different') if r['arm']=='stack2' else r for r in records]
        assert not paired_history(other_recipe,[])
        latest=make_latest(records,[],manifest,[],base)
        row=next(r for r in latest if r['name']=='stack2_s42')
        assert row['primary_epoch']==50 and row['fid_cfg1'] is None and row['latest_any_eval_epoch']==100
        checks+=['CSV precedence, rollback/stale-log suppression, matched-epoch contrasts and no mixing CFG epochs']
        wrong=dict(manifest[1]['config'],rep_multiplier=1.)
        (root/'config.json').write_text(json.dumps(wrong))
        cev,_,canonical,_,invalid,_=read_result_csvs(base,manifest,[])
        records=merge_rows(left+right+cev,[],canonical,invalid)
        assert not paired_history(records,[])
        (root/'config.json').write_text(json.dumps(manifest[1]['config']))
        checks+=['Mismatched run configuration quarantines CSVs and older logs']
        altered=dict(right[0],fid_generated=3.9,source_file='another_attempt.out')
        conflicts=merge_rows([right[0],altered],[])
        assert not conflicts[0]['active']
        checks+=['Conflicting log-only attempts cannot enter scientific comparisons']
        many=[]
        for seed in SEEDS:
            meta_a=meta_for((seed-42)*4+1,manifest);meta_b=meta_for((seed-42)*4,manifest)
            for epoch in ([50,100] if seed in (42,43) else [50]):
                for meta,fid in [(meta_a,4.),(meta_b,5.)]:
                    many.append(normalize_eval(evaluation(epoch=epoch,fid=fid),meta,path,1,'log_json_full_precision',[]))
        history=paired_history(merge_rows(many,[]),[]);summaries=seed_summaries(history,manifest)
        relevant=[r for r in summaries if r['comparison']=='stack2-minus-flat' and r['cfg']==3 and r['metric']=='a_minus_b']
        assert {r['n'] for r in relevant}=={3,5} and all(r['epoch']==50 and r['mean']==-1 for r in relevant)
        assert all(r['n']==0 for r in seed_summaries(history,manifest,endpoint=True))
        checks+=['Five-seed and fresh-three-seed summaries use a common epoch; endpoint stays fixed at 500']
        bad=normalize_eval(evaluation(fid=float('nan')),meta_for(1,manifest),path,1,'log_json_full_precision',[])
        assert not primary(merge_rows([bad],[])[0])
        bad2=normalize_eval(evaluation(fid=-1.),meta_for(1,manifest),path,1,'log_json_full_precision',[])
        assert not primary(merge_rows([bad2],[])[0])
        (base/'slurm_logs'/'fmcap_c0_1090160.err').write_text('Traceback (most recent call last):\nCUDA out of memory\n')
        checks+=['Invalid FIDs excluded; .err traceback/OOM detected']
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(['--base-dir',str(base)])==0
        out=base/'live_compiled'
        assert (out/'REPORT.md').exists() and (out/'live_fmnist_snapshot.zip').exists()
        with zipfile.ZipFile(out/'live_fmnist_snapshot.zip') as archive:assert archive.testzip() is None
        assert all(p.stat().st_size for p in out.glob('*.csv'))
        notes=json.loads((out/'scan_notes.json').read_text())
        assert any(s['error_lines'] for s in notes['logs'])
        with contextlib.redirect_stdout(io.StringIO()):assert main(['--base-dir',str(base),'--logs-only','--no-zip'])==0
        (base/'manifest.json').unlink()
        assert len(load_manifest(base,[]))==20
        checks+=['Complete CLI/ZIP, logs-only operation, no completed jobs required, fallback design']
    print(json.dumps(dict(passed=True,checks=checks),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
