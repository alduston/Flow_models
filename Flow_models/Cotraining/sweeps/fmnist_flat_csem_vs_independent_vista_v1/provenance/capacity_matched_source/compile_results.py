"""Compile partial or complete CSVs; never pair different epochs/protocols."""
from __future__ import annotations
import csv
import json
import math
from datetime import datetime,timezone
from pathlib import Path
import statistics
from experiments import rows, SEEDS, CONFIRMATION_SEEDS

PAIRS = (
    ('stack2-minus-flat','stack2','flat','primary: exact architecture and capacity'),
    ('stack3-minus-flat-three-encoder','stack3','flat_three_encoder','primary: exact architecture and capacity'),
    ('stack3-minus-flat','stack3','flat','secondary: matched parameter budget; encoder allocation differs'),
    ('stack3-minus-stack2','stack3','stack2','secondary: matched parameter budget; encoder allocation differs'),
)
EVAL_FIELDS = ['cell_id','name','arm','seed','stochastic_blocks','epoch','cfg','steps','init',
               'num_eval','backend','fid_generated','fid_posterior_recon']
PAIR_FIELDS = ['comparison','role','seed','epoch','cfg','steps','init','num_eval','backend',
               'arm_a','arm_b','fid_a','fid_b','a_minus_b','recon_a','recon_b','recon_a_minus_b']


def read_csv(path):
    if not path.exists():return []
    with path.open(newline='') as f:return list(csv.DictReader(f))


def write_csv(path,values,fields):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(values)


def finite(value):
    result=float(value)
    if not math.isfinite(result):raise ValueError('Nonfinite FID')
    return result


def stats(values):
    n=len(values)
    out=dict(n=n,mean=statistics.mean(values) if n else None,sd=None,se=None,ci95_low=None,ci95_high=None)
    if n>=2:
        from scipy.stats import t
        sd=statistics.stdev(values);se=sd/math.sqrt(n);radius=float(t.ppf(.975,n-1))*se
        out.update(sd=sd,se=se,ci95_low=out['mean']-radius,ci95_high=out['mean']+radius)
    return out


def protocol_key(r):
    return (int(r['epoch']),float(r['cfg']),int(r['steps']),r['init'],int(r['num_eval']),r['backend'])


def run(base,strict=False):
    base=Path(base);dest=base/'compiled';dest.mkdir(parents=True,exist_ok=True)
    history=[];statuses=[];errors=[];indices={}
    for cell in rows():
        config=cell['config'];out=base/'results'/cell['name']
        loss=read_csv(out/'loss.csv');last=max([int(r['epoch']) for r in loss],default=0)
        done=False
        if (out/'complete.json').exists():
            try:
                from paper import validate_complete
                validate_complete(cell,out);done=True
            except Exception as exc:errors.append(f'{cell["name"]}: completion check: {exc}')
        statuses.append(dict(cell_id=cell['cell_id'],name=cell['name'],arm=config['arm'],seed=config['seed'],
            training_epoch=last,complete=done,parameter_audit_present=(out/'parameter_audit.json').exists()))
        cache={}
        if (out/'eval.csv').exists():
            try:
                actual=json.loads((out/'config.json').read_text())
                mismatches=[k for k,v in config.items() if actual.get(k)!=v]
                if mismatches:raise ValueError('configuration mismatch: '+', '.join(mismatches))
                guard=out/'resume_recipe.json'
                if guard.exists():
                    from experiments import digest
                    if json.loads(guard.read_text())['source_hashes']!=digest(base):
                        raise ValueError('source hashes differ from the run')
            except (OSError,ValueError,KeyError) as exc:
                errors.append(f'{cell["name"]}: refusing evaluation input: {exc}')
                indices[(config['arm'],config['seed'])]=cache
                continue
        for row in read_csv(out/'eval.csv'):
            try:
                if None in row.values() or None in row:continue  # currently incomplete CSV line
                normalized={k:row[k] for k in ('epoch','cfg','steps','init','num_eval','backend')}
                normalized.update(epoch=int(float(row['epoch'])),cfg=float(row['cfg']),steps=int(row['steps']),
                    num_eval=int(row['num_eval']),fid_generated=finite(row['fid_generated']),
                    fid_posterior_recon=finite(row['fid_posterior_recon']),cell_id=cell['cell_id'],
                    name=cell['name'],arm=config['arm'],seed=config['seed'],stochastic_blocks=config['stochastic_blocks'])
                key=protocol_key(normalized)
                if key in cache:raise ValueError(f'duplicate evaluation key {key}')
                cache[key]=normalized;history.append(normalized)
            except (ValueError,KeyError,TypeError) as exc:
                errors.append(f'{cell["name"]}: {exc}')
        indices[(config['arm'],config['seed'])]=cache
    write_csv(dest/'eval_history.csv',history,EVAL_FIELDS)
    write_csv(dest/'status.csv',statuses,list(statuses[0]))
    primary=lambda key:key[1] in (1.,3.) and key[2:]==(20,'gaussian',10000,'inception-legacy')
    latest=[]
    for cache in indices.values():
        candidates=[r for k,r in cache.items() if primary(k) and k[1]==3.]
        if candidates:latest.append(max(candidates,key=lambda r:r['epoch']))
    write_csv(dest/'latest.csv',latest,EVAL_FIELDS)
    paired=[]
    for label,arm_a,arm_b,role in PAIRS:
        for seed in SEEDS:
            aa=indices[(arm_a,seed)];bb=indices[(arm_b,seed)]
            for key in sorted(aa.keys() & bb.keys()):
                if not primary(key):continue
                a,b=aa[key],bb[key]
                paired.append(dict(comparison=label,role=role,seed=seed,epoch=key[0],cfg=key[1],steps=key[2],
                    init=key[3],num_eval=key[4],backend=key[5],arm_a=arm_a,arm_b=arm_b,
                    fid_a=a['fid_generated'],fid_b=b['fid_generated'],a_minus_b=a['fid_generated']-b['fid_generated'],
                    recon_a=a['fid_posterior_recon'],recon_b=b['fid_posterior_recon'],
                    recon_a_minus_b=a['fid_posterior_recon']-b['fid_posterior_recon']))
    write_csv(dest/'paired_history.csv',paired,PAIR_FIELDS)
    latest_pairs={}
    for p in paired:
        key=(p['comparison'],p['seed'],p['cfg'])
        if key not in latest_pairs or p['epoch']>latest_pairs[key]['epoch']:latest_pairs[key]=p
    write_csv(dest/'paired_latest.csv',list(latest_pairs.values()),PAIR_FIELDS)
    arm_summaries=[];pair_summaries=[]
    for cohort,seeds in [('all_five',SEEDS),('new_confirmation',CONFIRMATION_SEEDS)]:
        for arm,_,_,_,_ in __import__('experiments').ARMS:
            for cfg in (1.,3.):
                wanted=(500,cfg,20,'gaussian',10000,'inception-legacy')
                values=[indices[(arm,s)][wanted] for s in seeds if wanted in indices[(arm,s)]]
                for metric in ('fid_generated','fid_posterior_recon'):
                    arm_summaries.append(dict(cohort=cohort,arm=arm,epoch=500,cfg=cfg,metric=metric,
                        expected_n=len(seeds),complete=len(values)==len(seeds),**stats([v[metric] for v in values])))
        for label,_,_,role in PAIRS:
            for cfg in (1.,3.):
                values=[p for p in paired if p['comparison']==label and p['seed'] in seeds and p['epoch']==500 and p['cfg']==cfg]
                for metric in ('a_minus_b','recon_a_minus_b'):
                    pair_summaries.append(dict(cohort=cohort,comparison=label,role=role,epoch=500,cfg=cfg,metric=metric,
                        expected_n=len(seeds),complete=len(values)==len(seeds),**stats([v[metric] for v in values])))
    write_csv(dest/'endpoint_arms.csv',arm_summaries,list(arm_summaries[0]))
    write_csv(dest/'endpoint_pairs.csv',pair_summaries,list(pair_summaries[0]))
    summary=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),expected_jobs=len(rows()),
        completed_jobs=sum(r['complete'] for r in statuses),historical_baseline=json.loads((base/'provenance'/'baseline_4p3.json').read_text()),
        primary='epoch 500, Gaussian T2 start, temperature 1, CFG3, RK4-20, 10000 legacy Inception samples',
        uncertainty='Student-t intervals across paired training seeds; not FID-estimator confidence intervals. Pointwise, no multiplicity correction.',
        seed_selection='42/43 were used to choose r=.5 historically. Report 44/45/46 separately as new confirmation.',
        interpretation='Negative A-minus-B favors A. No best-checkpoint, per-arm CFG, or best-seed selection. No numerical experiment establishes universal superiority.',
        endpoint_arms=arm_summaries,endpoint_pairs=pair_summaries,errors=errors)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Matched FMNIST comparison', '', f'Snapshot UTC: {summary["snapshot_utc"]}', '',
        f'Validated complete jobs: {summary["completed_jobs"]}/{summary["expected_jobs"]}.', '',
        summary['primary']+'.', '',summary['uncertainty'], '',summary['seed_selection'], '',
        'Every row below pairs the same epoch, CFG, solver, initialization, sample count, and feature backend. A negative delta favors the first arm. Live rows are provisional.', '',
        '| Comparison | Seed | Epoch | CFG | FID A | FID B | A - B |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in latest_pairs.values():
        if p['cfg']==3.:lines.append(f'| {p["comparison"]} | {p["seed"]} | {p["epoch"]} | {p["cfg"]:g} | {p["fid_a"]:.4f} | {p["fid_b"]:.4f} | {p["a_minus_b"]:+.4f} |')
    lines += ['', '## Prespecified endpoint comparisons', '',
        '| Cohort | Comparison | n / expected | Mean paired delta | 95% interval |', '|---|---|---:|---:|---|']
    for p in pair_summaries:
        if p['cfg']!=3. or p['metric']!='a_minus_b' or not p['n']:continue
        interval='unavailable' if p['ci95_low'] is None else f'[{p["ci95_low"]:+.4f}, {p["ci95_high"]:+.4f}]'
        lines.append(f'| {p["cohort"]} | {p["comparison"]} | {p["n"]} / {p["expected_n"]} | {p["mean"]:+.4f} | {interval} |')
    lines += ['',summary['interpretation'], '', 'Historical 4.304 is a reproduction target and is never inserted as a fresh run.']
    if errors:lines+=['','Input/completion errors:']+[f'- {e}' for e in errors]
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'[REPORT] {dest / "report.md"}; complete {summary["completed_jobs"]}/{len(rows())}; input errors {len(errors)}')
    return 1 if strict and (errors or summary['completed_jobs']!=len(rows())) else 0
