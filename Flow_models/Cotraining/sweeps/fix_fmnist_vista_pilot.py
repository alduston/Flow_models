#!/usr/bin/env python3
"""Fix the Vista FMNIST pilot's reload check before production starts.

Upload this file to the parent sweeps directory, enter the comparison bundle,
and run: python ../fix_fmnist_vista_pilot.py
No cluster job is submitted by this installer. A failed pilot is archived,
never relabelled as a completed run. Full training/evaluation are unchanged.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import fcntl
import getpass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

META={'evaluate_old': 'a7e61e06177e7807b0b073916eb6b7bfe52f1e373693df72213b7afb44fcfb13', 'evaluate_new': 'a4bfe4087106dd74d0005a20249ba96b451d4b1e45c402c0e5acb29588416d59', 'checks_old': 'd0bec4be1e8bca1316d51ac8ed3b03891c467fc4356e082cbee08c36a87979b7', 'checks_new': '555850621d968e30df62bf9eef481312ac62355cd1064d733f5ccf9eaca1969d'}
OLD_PARITY='@torch.no_grad()\ndef reload_parity(args,vae,head,path,device):\n    """Same saved inputs: reload, entire reverse solve, decoder and encoder moments."""\n    from train_stages import state_hash\n    with old.preserved_state(vae,head):\n        vae.eval()\n        if head is not None:head.eval()\n        saved=torch.load(path,map_location=\'cpu\',weights_only=False)\n        v,h=models.build(args,device);v.load_state_dict(saved[\'vae\']);v.eval()\n        if head is not None:h.load_state_dict(saved[\'ema\']);h.eval()\n        assert state_hash(v.state_dict())==state_hash(vae.state_dict())\n        if head is not None:assert state_hash(h.state_dict())==state_hash(head.state_dict())\n        g=torch.Generator().manual_seed(args.eval_seed+99)\n        x=torch.randn(2,1,32,32,generator=g).to(device).clamp(-1,1)\n        eta=[torch.randn(2,*s,generator=g).to(device) for s in vae.shapes]\n        d1=vae.draw(x,innovations=eta);d2=v.draw(x,innovations=eta)\n        t=torch.full((2,),args.decode_time,device=device)\n        a=vae.decode(d1[\'z\'],t);b=v.decode(d2[\'z\'],t);err=float((a-b).abs().max())\n        if head is not None:\n            z=torch.randn(2,4,8,8,generator=g);labels=torch.tensor([0,9])\n            for cfg in (1.,3.):\n                z1=generate(args,head,z,labels,device,cfg);z2=generate(args,h,z,labels,device,cfg)\n                err=max(err,float((vae.decode(z1,t)-v.decode(z2,t)).abs().max()))\n        if err>1e-6:raise AssertionError(f\'Checkpoint-reload evaluation drift: {err}\')\n        return dict(passed=True,max_abs_error=err,inputs_seed=args.eval_seed+99,\n                    checks=[\'tensor_identity\',\'posterior_reconstruction\',\'RK4 generation at CFG1 and CFG3\'] if head is not None else [\'tensor_identity\',\'posterior_reconstruction\'])\n'
NEW_PARITY='@torch.no_grad()\ndef reload_parity(args,vae,head,path,device):\n    """Same saved inputs: reload, entire reverse solve, decoder and encoder moments."""\n    import copy\n    from train_stages import state_hash\n    original_device=str(device)\n    # Preserve live tensor values exactly in CPU copies; state hashes below\n    # compare those values against the saved snapshot before any inference.\n    # CPU replay avoids requiring identical CUDA/TF32 kernel arithmetic across\n    # separately allocated module instances. Training/evaluation stay on GPU.\n    if torch.device(device).type==\'cuda\':\n        vae=copy.deepcopy(vae).cpu()\n        if head is not None:head=copy.deepcopy(head).cpu()\n        device=torch.device(\'cpu\')\n    with old.preserved_state(vae,head):\n        vae.eval()\n        if head is not None:head.eval()\n        saved=torch.load(path,map_location=\'cpu\',weights_only=False)\n        v,h=models.build(args,device);v.load_state_dict(saved[\'vae\']);v.eval()\n        if head is not None:h.load_state_dict(saved[\'ema\']);h.eval()\n        assert state_hash(v.state_dict())==state_hash(vae.state_dict())\n        if head is not None:assert state_hash(h.state_dict())==state_hash(head.state_dict())\n        def peak_error(a,b):\n            value=float((a-b).abs().max())\n            if not math.isfinite(value):raise AssertionError(\'Nonfinite checkpoint-reload output\')\n            return value\n        g=torch.Generator().manual_seed(args.eval_seed+99)\n        x=torch.randn(2,1,32,32,generator=g).to(device).clamp(-1,1)\n        eta=[torch.randn(2,*s,generator=g).to(device) for s in vae.shapes]\n        d1=vae.draw(x,innovations=eta);d2=v.draw(x,innovations=eta)\n        t=torch.full((2,),args.decode_time,device=device)\n        a=vae.decode(d1[\'z\'],t);b=v.decode(d2[\'z\'],t);err=peak_error(a,b)\n        if head is not None:\n            z=torch.randn(2,4,8,8,generator=g);labels=torch.tensor([0,9])\n            for cfg in (1.,3.):\n                z1=generate(args,head,z,labels,device,cfg);z2=generate(args,h,z,labels,device,cfg)\n                err=max(err,peak_error(vae.decode(z1,t),v.decode(z2,t)))\n        if not math.isfinite(err) or err>1e-6:raise AssertionError(f\'Checkpoint-reload evaluation drift: {err}\')\n        return dict(passed=True,max_abs_error=err,inputs_seed=args.eval_seed+99,\n                    parity_device=str(device),original_device=original_device,absolute_tolerance=1e-6,\n                    checks=[\'tensor_identity\',\'posterior_reconstruction\',\'RK4 generation at CFG1 and CFG3\'] if head is not None else [\'tensor_identity\',\'posterior_reconstruction\'])\n'
OLD_CHECK="if not models_only:result['resume_checks']=resume_checks(base,device)"
NEW_CHECK="if not models_only:\n        result['resume_checks']=resume_checks(base,'cpu')\n        result['resume_check_device']='cpu'"
PILOT_NAMES={'pilot_joint','pilot_vae_pretrain','pilot_prior_dsm','pilot_prior_csem'}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise RuntimeError(message)
def field(text,name):
    found=re.search(r'(?:^|\s)'+name+r'=([^\s]+)',text)
    return found[1] if found else None

def install(base):
    base=base.resolve()
    require((base/'protocol.py').is_file() and (base/'tools/submit_vista.py').is_file(),
            'Enter the original Vista comparison bundle directory first')
    require(Path(__file__).resolve().parent!=base,
            'Keep this installer in the parent sweeps directory, not alongside root training sources')
    sys.path.insert(0,str(base))
    import protocol,campaign
    status=base/'status';status.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for name in ('watcher.lock','submit.lock'):
            handle=stack.enter_context((status/name).open('a'))
            try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('A watcher/submission process is active; let it exit before patching')
        marker=status/'pilot_reload_fix.json'
        if marker.exists() and digest(base/'evaluate.py')==META['evaluate_new'] and digest(base/'checks.py')==META['checks_new']:
            require(protocol.sources(base)==protocol.read(base/'source_hashes.json'),'Source manifest changed after this patch')
            print('[ALREADY APPLIED] No files or checkpoints changed.');return
        require(not marker.exists(),'Incomplete/changed prior patch; inspect status/pilot_reload_fix.json')
        require(digest(base/'evaluate.py')==META['evaluate_old'],'Unexpected evaluate.py; no patch applied')
        require(digest(base/'checks.py') in (META['checks_old'],META['checks_new']),
                'Unexpected checks.py; no patch applied')
        current=protocol.sources(base)
        require(current==protocol.read(base/'source_hashes.json'),
                'Source manifest does not match current files; no identities rewritten')
        ledger_path=status/'submissions.json'
        ledger=protocol.read(ledger_path) if ledger_path.exists() else {}
        require(set(ledger)<={'pilot'},'Production jobs already recorded; this pilot-only migration cannot be used')
        if ledger.get('pilot'):
            require(ledger['pilot']['signature']==campaign.signature(dict(name='pilot',kind='pilot',deps=[]),base),
                    'Pilot source signature mismatch; inspect before migrating')
        require(not (base/'pilot/timing.json').exists(),'Pilot already completed; do not invalidate existing timing')
        require(not (base/'selections').exists(),'Selections exist; this patch is only for an unstarted production campaign')
        result_root=base/'results'
        candidates=list(result_root.iterdir()) if result_root.exists() else []
        require(all(p.name in PILOT_NAMES for p in candidates),
                'Non-pilot results found; preserve their source/checkpoint identities')
        queue=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%j'],
                             check=True,capture_output=True,text=True)
        for line in queue.stdout.splitlines():
            pieces=line.strip().split('|')
            if len(pieces)!=2 or not pieces[1].startswith('fm3_'):continue
            jid=pieces[0].strip()
            detail=subprocess.run(['scontrol','show','job','-o',jid],check=True,capture_output=True,text=True).stdout
            directory=field(detail,'WorkDir')
            require(not directory or Path(directory).resolve()!=base,
                    'A job from this bundle is still active; no source changes allowed')
        old=(base/'evaluate.py').read_text();updated=old.replace(OLD_PARITY,NEW_PARITY)
        require(old.count(OLD_PARITY)==1,'Reload function did not match exactly')
        check_text=(base/'checks.py').read_text()
        if OLD_CHECK in check_text:check_text=check_text.replace(OLD_CHECK,NEW_CHECK)
        require(NEW_CHECK in check_text,'CPU resume check not present')
        compile(updated,'evaluate.py','exec');compile(check_text,'checks.py','exec')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup=base/'pilot_failed_attempts'/stamp;backup.mkdir(parents=True)
        for name in ('evaluate.py','checks.py','source_hashes.json','SHA256SUMS'):
            if (base/name).exists():shutil.copy2(base/name,backup/name)
        for name in ('submissions.json','pilot.json','watcher_error.json'):
            if (status/name).exists():shutil.copy2(status/name,backup/name)
        protocol.atomic(backup/'migration.json',dict(state='prepared',old_sources=current,old_ledger=ledger))
        # The pilot has only ten minibatches per stage; rerun it under the new
        # validation identity rather than rewriting tensors/checkpoint recipes.
        for p in candidates:
            dest=backup/'results'/p.name;dest.parent.mkdir(exist_ok=True);shutil.move(str(p),str(dest))
        if (base/'pilot').exists():shutil.move(str(base/'pilot'),str(backup/'pilot'))
        # Every precondition is checked before changing the active source files.
        (base/'evaluate.py').write_text(updated)
        (base/'checks.py').write_text(check_text)
        protocol.atomic(base/'source_hashes.json',protocol.sources(base))
        if ledger.get('pilot'):
            ledger['pilot']['signature']=campaign.signature(dict(name='pilot',kind='pilot',deps=[]),base)
            ledger['pilot']['validation_fix_backup']=str(backup)
            protocol.atomic(ledger_path,ledger)
        sums=base/'SHA256SUMS'
        if sums.exists():
            lines=[]
            for line in sums.read_text().splitlines():
                value,name=line.split('  ',1)
                if name in ('evaluate.py','checks.py','source_hashes.json'):value=digest(base/name)
                lines.append(value+'  '+name)
            sums.write_text('\n'.join(lines)+'\n')
        protocol.atomic(marker,dict(applied=True,backup=str(backup),new_sources=protocol.sources(base),
            change='Exact tensor identity plus strict CPU inference replay at1e-6; GPU training/evaluation unchanged',
            old_checkpoint_recipes_rewritten=False))
        print('[FIXED] Strict reload inference now runs on CPU copies; real training/evaluation remain on CUDA.')
        print('[ARCHIVED]',backup)
        print('[NEXT] python tools/submit_vista.py --phase pilot --account DMS22021')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=Path.cwd())
    install(parser.parse_args().base_dir)
