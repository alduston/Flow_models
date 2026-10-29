"""Exactly solved stacked-teacher bias/curl in linear Gaussian hierarchies."""
import csv
import math
from pathlib import Path
import numpy as np


def matrices(beta,variance,t):
    beta=np.asarray(beta,float);variance=np.asarray(variance,float);d=len(variance)
    transform=np.linalg.inv(np.eye(d)-beta)
    covariance=transform@np.diag(variance)@transform.T
    alpha=math.exp(-t);sigma=math.sqrt(-math.expm1(-2*t))
    noisy=alpha**2*covariance+sigma**2*np.eye(d)
    posterior=alpha*covariance@np.linalg.inv(noisy)
    denominators=alpha**2*variance+sigma**2
    stack=(sigma/denominators)[:,None]*(np.eye(d)-alpha*beta@posterior)
    valid=sigma*np.linalg.inv(noisy)
    return covariance,noisy,stack,valid


def reverse_covariance(beta,variance,teacher='naive',initialization='gaussian',t_min=2e-5,t_max=2.):
    """Ideal learned linear field, exact probability-flow covariance evolution.

    Forward OU PF drift is -y-score = (-I+B_t/sigma_t)y. Integrate its
    covariance equation backward in time, with no network/regression error.
    """
    from scipy.integrate import solve_ivp
    covariance,noisy,_,_=matrices(beta,variance,t_max)
    d=len(variance);eye=np.eye(d)
    start=eye if initialization=='gaussian' else noisy
    def rhs(t,vector):
        _,_,b,m=matrices(beta,variance,t)
        sigma=math.sqrt(-math.expm1(-2*t))
        drift=-eye+(b if teacher=='naive' else m)/sigma
        p=vector.reshape(d,d)
        return (drift@p+p@drift.T).reshape(-1)
    result=solve_ivp(rhs,(t_max,t_min),start.reshape(-1),rtol=2e-9,atol=1e-11)
    if not result.success:raise RuntimeError(result.message)
    p=result.y[:,-1].reshape(d,d);p=(p+p.T)/2
    target=matrices(beta,variance,t_min)[1]
    return p,target


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];sampling=[]
    for blocks in (2,3):
        for topology in ('full','markov','siblings','decoupled') if blocks==3 else ('full','decoupled'):
            for coupling in (0.,.1,.25,.5,1.,2.):
                beta=np.zeros((blocks,blocks));beta[1,0]=coupling
                if blocks==3:
                    if topology in ('full','siblings'):beta[2,0]=.6*coupling
                    if topology in ('full','markov'):beta[2,1]=coupling
                if topology=='decoupled':beta[:]=0
                for t in (2e-5,.001,.01,.05,.2,.5,1.,2.):
                    _,s,b,m=matrices(beta,[.8]*blocks,t)
                    diff=b-m;error=np.diag(diff@s@diff.T);energy=np.diag(m@s@m.T)
                    for i in range(blocks):rows.append(dict(blocks=blocks,topology=topology,coupling=coupling,time=t,
                        block=i+1,bias_squared=error[i],true_epsilon_energy=energy[i],
                        relative_bias_rms=math.sqrt(max(0,error[i]/energy[i])),curl_energy=float(np.square(b-b.T).sum())))
                if coupling in (0.,.25,1.):
                    for teacher in ('valid','naive'):
                        for initialization in ('gaussian','oracle_qT'):
                            p,target=reverse_covariance(beta,[.8]*blocks,teacher,initialization)
                            sampling.append(dict(blocks=blocks,topology=topology,coupling=coupling,teacher=teacher,initialization=initialization,
                                relative_covariance_error=float(np.linalg.norm(p-target)/np.linalg.norm(target)),
                                min_generated_eigenvalue=float(np.linalg.eigvalsh(p).min())))
    with (out/'gaussian_exact.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    with (out/'gaussian_reverse_covariance.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(sampling[0]));wr.writeheader();wr.writerows(sampling)
    return rows


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='report/gaussian_sandbox')
    a=p.parse_args();run(a.output)
