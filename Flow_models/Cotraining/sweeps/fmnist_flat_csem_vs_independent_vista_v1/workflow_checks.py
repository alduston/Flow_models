"""Offline scientific/export and scheduler tests; fixtures never enter results/."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from protocol import ROOT,atomic,rows,common,resolve,sources,REGIMES,FRESH
import campaign
from compile_results import compile,stats,write_csv

def scheduler_check():
    with tempfile.TemporaryDirectory() as temp:
        base=Path(temp);(base/'pinned').mkdir();(base/'provenance').mkdir()
        (base/'job.slurm').write_text('fixture')
        (base/'provenance/flat_three_encoder_cell2.json').write_bytes((ROOT/'provenance/flat_three_encoder_cell2.json').read_bytes())
        atomic(base/'pilot/timing.json',dict(gpu='offline fixture',stages={s:dict(seconds_per_batch=.1,evaluation_seconds=1.) for s in ('joint','vae_pretrain','prior_dsm','prior_csem')}))
        calls=[];ids={};active=False
        def fake(cmd,**kwargs):
            nonlocal active
            if cmd[0]=='squeue':
                output='\n'.join(f'{j}|PENDING|Priority' for j in ids.values()) if active else ''
                return __import__('subprocess').CompletedProcess(cmd,0,stdout=output,stderr='')
            assert cmd[0]=='sbatch'
            calls.append(cmd);jid=str(1000+len(calls))
            name=next(v.split('=',1)[1][4:] for v in cmd if v.startswith('--job-name='))
            ids[name]=jid
            return __import__('subprocess').CompletedProcess(cmd,0,stdout=jid+'\n',stderr='')
        with patch.object(campaign.subprocess,'run',side_effect=fake),patch.object(campaign,'task_complete',return_value=False),redirect_stdout(io.StringIO()):
            campaign.submit(base,phase='all',time_limit='auto')
            assert len(calls)==48
            rr=rows(base)
            for r,cmd in zip(rr,calls):
                actual=[x.split('=',1)[1] for x in cmd if x.startswith('--dependency=')]
                expected='afterok:'+':'.join(ids[d] for d in r['deps']) if r['deps'] else None
                assert actual==([expected] if expected else [])
                assert '--partition=gh' in cmd
            active=True;campaign.submit(base,phase='all',time_limit='auto');assert len(calls)==48
        return dict(training_tasks=45,slurm_jobs=48,dependencies='all exact afterok parent IDs',repeat_submission='no duplicates')

def compiler_check():
    with tempfile.TemporaryDirectory() as temp:
        base=Path(temp);(base/'provenance').mkdir()
        (base/'provenance/flat_three_encoder_cell2.json').write_bytes((ROOT/'provenance/flat_three_encoder_cell2.json').read_bytes())
        atomic(base/'selections/final.json',dict(beta0=.03,lr_prior=2e-4/.6,fresh_seeds=list(FRESH)))
        atomic(base/'selections/shortlist.json',dict(betas=[.01,.03]))
        for r in rows(base):
            if r['kind']!='train' or r['group']!='final':continue
            a=resolve(r,base);out=base/'results'/r['name'];out.mkdir(parents=True)
            cfg=dict(vars(a),synthetic=False,source_sha256=f'paired_source_seed{a.seed}',initial_head_sha256=f'paired_head_seed{a.seed}',evaluation_contract={'test_fixture':True})
            atomic(out/'config.json',cfg);atomic(out/'complete.json',dict(snapshot_sha256=f'paired_source_seed{a.seed}',timing=dict(training_seconds=7200.,evaluation_seconds=3.,optimizer_steps=100.,data_passes=500.,examples_seen=1000.,source_training_seconds=3600. if a.stage.startswith('prior') else 0.,allocation_seconds=7205.)))
            metrics=[]
            for epoch in range(50,501,50):
                for g in ([0.] if a.stage=='vae_pretrain' else [1.,3.]):
                    # A transparent fake offset lets the paired estimator result be known exactly.
                    offset={'cotrained_csem':0.,'indep_tweedie':2.,'indep_csem':1.,'independent_vae':3.}[a.regime]
                    metrics.append(dict(epoch=epoch,seed=a.seed,regime=a.regime,cfg=g,kind='reconstruction' if a.stage=='vae_pretrain' else 'generation',
                        eval_split='test',backend='inception-legacy',steps=20,num_eval=10000,temperature=1.,fid_generated=10.-epoch/100+offset+(a.seed-49)*.1,
                        fid_posterior_recon=2.,kid=.001 if epoch==500 else '',kid_subset_sd=.0001,
                        training_seconds=epoch*12.,source_training_seconds=3600. if a.stage.startswith('prior') else 0.,
                        end_to_end_training_gpu_hours=(epoch*12.+(3600. if a.stage.startswith('prior') else 0.))/3600))
            write_csv(out/'eval.csv',metrics)
        with patch('compile_results.validate_complete',side_effect=lambda out,production=True:json.loads((Path(out)/'complete.json').read_text())),redirect_stdout(io.StringIO()):
            compile(base,strict=True)
            contrasts=campaign.csv_rows(base/'compiled/paired_differences.csv')
            assert float(contrasts[0]['mean'])==-1. and float(contrasts[0]['sd'])<1e-12
            summary=campaign.csv_rows(base/'compiled/curve_summary.csv');assert len(summary)==30 and {r['n'] for r in summary}=={'5'}
            assert (base/'compiled/fmnist_comparison.pdf').stat().st_size>1000
            path=base/'results/final_joint_s47/eval.csv';text=path.read_text();path.write_text(text+text.splitlines()[1]+'\n')
            try:compile(base,strict=True)
            except ValueError:pass
            else:raise AssertionError('Duplicate metric row accepted')
            assert not (base/'compiled/complete.json').exists()
            path.write_text(text)
            (base/'results/final_joint_s47/complete.json').unlink()
            try:compile(base,strict=True)
            except ValueError:pass
            else:raise AssertionError('Incomplete seed accepted')
        return dict(known_paired_difference=-1.,strict_seed_count=5,duplicate_rejection=True,missing_seed_rejection=True,plots='PDF/SVG/PNG generated from isolated fixtures')

def selection_check():
    with tempfile.TemporaryDirectory() as temp:
        base=Path(temp);(base/'provenance').mkdir();(base/'pinned').mkdir();(base/'job.slurm').write_text('fixture')
        (base/'provenance/flat_three_encoder_cell2.json').write_bytes((ROOT/'provenance/flat_three_encoder_cell2.json').read_bytes())
        rec={0.:2.2,.01:2.6,.03:2.85,.07:3.3,.15:4.,.30:6.}
        for beta in campaign.BETAS:
            out=base/'results'/f'screen_vae_b{beta:g}_s42'
            atomic(out/'complete.json',dict(recipe_sha256=f'fixture_beta{beta}'))
            atomic(out/'diagnostics_epoch_0500.json',dict(kl0=1./(beta+.001),posterior_variance=beta))
        def ep(out,cfg=3.):
            name=Path(out).name
            if name=='screen_joint_s42':return dict(fid_posterior_recon=2.8)
            if name.startswith('screen_vae'):
                beta=float(name.split('_b')[1].split('_')[0]);return dict(fid_posterior_recon=rec[beta])
            c=json.loads((Path(out)/'config.json').read_text())
            return dict(fid_generated=5.+abs(c['beta0']-.03)*10+int(c['lr_prior']>4e-4)+int(c['stage']=='prior_dsm')*.1)
        with patch.object(campaign,'endpoint',side_effect=ep),redirect_stdout(io.StringIO()):
            selected=campaign.shortlist(base)
            assert selected['betas']==[.01,.03]
            for row in rows(base):
                if row.get('group')=='development' and row.get('stage','').startswith('prior'):
                    a=resolve(row,base)
                    atomic(base/'results'/row['name']/'config.json',dict(vars(a),source_sha256=f'fixture_parent_b{a.beta0}_s{a.seed}',initial_head_sha256=f'head_s{a.seed}',split_metadata={'hash':'fixed_split'},recipe_sha256=row['name']))
            final=campaign.freeze(base)
            assert final['beta0']==.03 and final['lr_prior']==campaign.LRS[0]
            campaign.freeze(base)  # identical re-entry is permitted
            try:campaign.shortlist(base,[.03,.07])
            except ValueError:pass
            else:raise AssertionError('Frozen shortlist could be changed')
        return dict(shortlist='reconstruction match and frontier',common_recipe_selection='both objectives and both development seeds',immutability=True)

def run():
    expected=stats([1.,2.,3.,4.,5.]);assert expected['mean']==3. and abs(expected['sd']-np.sqrt(2.5))<1e-14
    result=dict(scheduler=scheduler_check(),selection=selection_check(),compiler=compiler_check(),statistics=expected,fixture_metrics='temporary only, never production')
    atomic(ROOT/'validation/workflow_checks.json',result);print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(run())
