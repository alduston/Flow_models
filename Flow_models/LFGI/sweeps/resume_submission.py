#!/usr/bin/env python3
"""Finish an interrupted submission DAG using an already accepted setup job.
Standalone, stdlib-only: add this file to the existing bundle without changing
its frozen files or interrupting setup. Does not run numerical work on login.
"""
import argparse,fcntl,json,os,re,subprocess,sys,time
from pathlib import Path

def jobid(text):
 ids=[]
 for line in text.splitlines():
  m=re.fullmatch(r'(?:Submitted batch job\s+)?([0-9]+)(?:;[A-Za-z0-9_.-]+)?',line.strip())
  if m:ids.append(m.group(1))
 if len(set(ids))!=1:raise RuntimeError('Cannot uniquely parse accepted job output. It may already be queued; inspect logs/submission_responses.log and do not blindly retry.')
 return ids[-1]
def command(args):return subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
def info(id):
 r=command(['scontrol','show','job',str(id),'-o'])
 if r.returncode==0 and re.search(r'\bJobState=',r.stdout):
  def value(k):
   m=re.search(r'\b'+k+r'=(\S+)',r.stdout);return m.group(1) if m else ''
  return dict(name=value('JobName'),state=value('JobState'),workdir=value('WorkDir'))
 r=command(['sacct','-n','-X','-j',str(id),'--format=JobIDRaw,JobName%80,State%32,WorkDir%512','--parsable2'])
 if r.returncode!=0:raise RuntimeError('Cannot verify job '+str(id)+': '+r.stderr)
 for line in r.stdout.splitlines():
  p=[s.strip() for s in line.split('|')]
  if len(p)>=4 and p[0]==str(id):return dict(name=p[1],state=p[2].split()[0].rstrip('+'),workdir=p[3])
 raise RuntimeError('No scheduler/accounting record for job '+str(id))
def atomic(path,data):
 p=path.with_name(path.name+'.tmp.'+str(os.getpid()));p.write_text(json.dumps(data,indent=2));os.replace(p,path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--setup',required=True,type=int);p.add_argument('--pilot',action='store_true');p.add_argument('--with-reference',action='store_true');p.add_argument('--no-marginals',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parent
 if not (root/'ls6/job.sh').exists():raise RuntimeError('Place resume_submission.py in the extracted LFGI_Quake_LS6_CPU_Array directory')
 os.chdir(root);logs=root/'logs';logs.mkdir(exist_ok=True);lock=(root/'.submission.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 setup=str(a.setup);s=info(setup)
 if s['name']!='quake-setup':raise RuntimeError('Selected job is not quake-setup: '+str(s))
 if s['workdir'] and Path(s['workdir']).resolve()!=root:raise RuntimeError('Setup belongs to a different work directory: '+s['workdir'])
 if s['state'] not in ['PENDING','RUNNING','CONFIGURING','COMPLETING','COMPLETED']:raise RuntimeError('Setup did not complete successfully or is not runnable: '+str(s))
 profile='pilot' if a.pilot else 'production';binding=dict(setup=setup,root=str(root),profile=profile,reference=a.with_reference,marginals=not a.no_marginals);statefile=logs/'recovery_submission.json'
 records={}
 idsfile=logs/'submitted_ids.txt'
 if idsfile.exists():
  for line in idsfile.read_text().splitlines():
   q=line.split()
   if len(q)==2:records[q[0]]=q[1]
 if records.get('setup',setup)!=setup:raise RuntimeError('Recorded setup differs; refusing to mix submission graphs')
 if statefile.exists():
  saved=json.loads(statefile.read_text())
  if saved['binding']!=binding:raise RuntimeError('Recovery options differ from the previous attempt')
  tasks=saved['tasks'];cap=saved['concurrency']
 else:
  sys.path.insert(0,str(root/'ls6'));from limits import plan
  pre=plan(profile,a.with_reference,not a.no_marginals);tasks=pre['tasks'];cap=pre['concurrency'];atomic(logs/'recovery_preflight.json',pre);atomic(statefile,dict(binding=binding,tasks=tasks,concurrency=cap))
 records['setup']=setup
 def saveids():
  tmp=idsfile.with_name('submitted_ids.txt.tmp');tmp.write_text(''.join(k+' '+v+'\n' for k,v in records.items()));os.replace(tmp,idsfile)
 saveids()
 def submit(stage,wall,cpus,dependency=None,array=None):
  if stage in records:
   previous=info(records[stage])
   if previous['name']!='quake-'+stage or (previous['workdir'] and Path(previous['workdir']).resolve()!=root):raise RuntimeError('Recorded child does not match this campaign: '+str(previous))
   print('Reusing',stage,records[stage],previous['state'],flush=True);return records[stage]
  args=['sbatch','--parsable','-A',os.getenv('QUAKE_ACCOUNT','DMS22021'),'-p',os.getenv('QUAKE_QUEUE','normal'),'-N','1','-n','1','-c',str(cpus),'-t',wall,'-J','quake-'+stage]
  if array:args+=['--array='+array,'-o','logs/%x-%A_%a.out','-e','logs/%x-%A_%a.err']
  else:args+=['-o','logs/%x-%j.out','-e','logs/%x-%j.err']
  if dependency:args+=['--dependency='+dependency]
  args+=['ls6/job.sh',str(root),stage,profile];r=command(args)
  with (logs/'submission_responses.log').open('a') as f:f.write('\nRECOVERY '+stage+'\n'+r.stdout+'\nSTDERR\n'+r.stderr+'\n')
  if r.returncode:raise RuntimeError('sbatch failed for '+stage+': '+r.stderr)
  id=jobid(r.stdout);records[stage]=id;saveids();print('Submitted',stage,id,flush=True);return id
 # If setup has already finished and aged out of Slurm's controller, it needs no dependency.
 dep=None if s['state']=='COMPLETED' else 'afterok:'+setup
 smoke=submit('smoke',os.getenv('QUAKE_SMOKE_WALL','01:30:00'),64,dep)
 run=submit(profile,os.getenv('QUAKE_WALL','04:00:00'),128,'afterok:'+smoke,'0-%d%%%d'%(tasks-1,cap));parents=[run]
 if not a.no_marginals:parents.append(submit('marginals',os.getenv('QUAKE_MARGINALS_WALL','03:00:00'),64,'afterany:'+run))
 if a.with_reference:parents.append(submit('reference',os.getenv('QUAKE_REFERENCE_WALL','12:00:00'),1,'afterok:'+smoke))
 submit('report','00:30:00',1,'afterany:'+':'.join(parents));print('Recovery complete: reused setup '+setup+'; production array has '+str(tasks)+' tasks.');print(idsfile.read_text())
if __name__=='__main__':
 try:main()
 except Exception as e:print('Recovery stopped: '+str(e),file=sys.stderr);sys.exit(2)
