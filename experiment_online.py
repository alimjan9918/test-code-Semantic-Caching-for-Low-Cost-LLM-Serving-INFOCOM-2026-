"""Online semantic caching experiment corresponding to Fig. 2(c)."""
import argparse,csv,json,importlib.metadata
from pathlib import Path
import numpy as np
from experiment_known_parameters import loss,reverse_greedy,brute_force

def bounds(nc,sums,t,m,mode,T,delta):
 mean=np.divide(sums,nc,out=np.zeros(m),where=nc>0)
 r=np.sqrt(np.log(4*m*T**3/delta)/(2*np.maximum(nc,1)))
 z=mean+r if mode=='UCB' else mean-r if mode=='LCB' else mean
 z[nc==0]=1.0 if mode!='LCB' else 0.0
 return z

def run(p,c,D,k,T,seed,name,alpha=1,delta=.05,epsilon=.2):
 m=len(p); env=np.random.default_rng(seed); qstream=env.choice(m,T,p=p); noise=env.normal(0,.05,(T,m)); rng=np.random.default_rng(seed+99991)
 npq=np.zeros(m,int); nc=np.zeros(m,int); sums=np.zeros(m); cur=np.zeros(m,int); prev=np.zeros(m,int); pc=pp=0; M=[]; switches=recomp=0; fill=service=0.; rows=[]
 best_cache,opt=brute_force(p,c,D,k); static_lfu=np.argsort(-np.bincount(qstream,minlength=m),kind='stable')[:k].tolist()
 stages=[1]*m; mode='UCB' if name=='CUCB-SC' else 'LCB' if name in ('CLCB-SC','CLCB-SC-LS') else 'MEAN'
 for t,q in enumerate(qstream):
  update=name!='CLCB-SC-LS' or t==0
  if name=='CLCB-SC-LS' and t>0:
   eligible=np.flatnonzero(cur>=1+np.sqrt(T*np.maximum(prev,0)/m))
   if len(eligible):
    j=int(eligible[0]); prev[j]+=cur[j]; cur[j]=0; update=True
   elif pc>=1+np.sqrt(T*max(pp,0)): pp+=pc; pc=0; update=True
  if name=='LFU': update=t==0
  if update:
   phat=npq/t if t else np.full(m,1/m); est=bounds(nc,sums,t,m,mode,T,delta)
   if name=='LFU': new=static_lfu
   elif name=='Epsilon-Greedy':
    new=list(range(m))
    while len(new)>k:
     rem=int(rng.choice(new)) if rng.random()<epsilon else min(new,key=lambda x:(loss([j for j in new if j!=x],phat,est,D),x)); new.remove(rem)
   else: new=reverse_greedy(phat,est,D,k)
   recomp+=1; switches+=int(set(new)!=set(M)); fill+=sum(c[j] for j in set(new)-set(M)); M=list(new)
  dist=float(D[q,M].min())
  est=bounds(nc,sums,t,m,mode,T,delta)
  invoke=dist>0 and (nc[q]==0 or est[q]<dist)
  if invoke:
   observed=float(c[q]+noise[t,q]); service+=observed; nc[q]+=1; sums[q]+=observed; cur[q]+=1; expected=c[q]
  else: service+=dist; expected=dist
  npq[q]+=1; pc+=1; fill_now=fill
  if (t+1) in set(np.unique(np.linspace(1,T,min(T,100),dtype=int))): rows.append(dict(seed=seed,algorithm=name,t=t+1,service_cost=service,fill_cost=fill,average_excess_cost=(service+fill)/(t+1)-opt,average_excess_without_fill=service/(t+1)-opt,regret=(service+fill)-(t+1)*alpha*opt,cache_switches=switches,cache_recomputations=recomp))
 return rows

def main():
 root=Path(__file__).resolve().parent; ap=argparse.ArgumentParser()
 ap.add_argument('--input',type=Path,default=root/'fig2a_results'); ap.add_argument('--out',type=Path,default=root/'fig2c_results'); ap.add_argument('--rounds',type=int,default=5000); ap.add_argument('--seeds',type=int,default=10); ap.add_argument('--k',type=int,default=5); ap.add_argument('--delta',type=float,default=.05); ap.add_argument('--alpha',type=float,default=1.); args=ap.parse_args()
 with np.load(args.input/'embedding_and_distance.npz') as a:p,D=a['probabilities'],a['distances']
 costs=[]
 for s in range(args.seeds):
  with np.load(args.input/f'seed_{s:02d}_inputs.npz') as a: costs.append(a['costs'])
 args.out.mkdir(parents=True,exist_ok=True); rows=[]
 for s,c in enumerate(costs):
  for name in ['CUCB-SC','CLCB-SC','CLCB-SC-LS','Epsilon-Greedy','LFU']: rows.extend(run(p,c,D,args.k,args.rounds,s,name,args.alpha,args.delta))
 print('Completed seed',s+1,'/',args.seeds,flush=True)
 with (args.out/'online.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for name in ['CUCB-SC','CLCB-SC','CLCB-SC-LS','Epsilon-Greedy','LFU']:
  g=[r for r in rows if r['algorithm']==name and r['t']==args.rounds]; summary.append(dict(algorithm=name,mean_excess=float(np.mean([r['average_excess_cost'] for r in g])),mean_switches=float(np.mean([r['cache_switches'] for r in g])),mean_runtime='not measured'))
 with (args.out/'online_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7.5,4.8))
 for name,color in [('CLCB-SC','#1f77b4'),('CUCB-SC','#ff7f0e'),('CLCB-SC-LS','#2ca02c'),('Epsilon-Greedy','#d62728'),('LFU','#9467bd')]:
  g=[r for r in rows if r['algorithm']==name]; xs=sorted(set(r['t'] for r in g)); ys=[np.mean([r['average_excess_cost'] for r in g if r['t']==x]) for x in xs]; sd=[np.std([r['average_excess_cost'] for r in g if r['t']==x],ddof=1) for x in xs]; ax.plot(xs,ys,label=name,color=color);ax.fill_between(xs,np.array(ys)-sd,np.array(ys)+sd,color=color,alpha=.12)
 ax.set(xlabel='Rounds (t)',ylabel='Average excess cost',title='Online Semantic Caching: m=20, k=5');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(args.out/'fig2c_online.png',dpi=200);plt.close(fig)
 (args.out/'config.json').write_text(json.dumps(vars(args),default=str,indent=2),encoding='utf-8');print('Results saved to',args.out.resolve())
if __name__=='__main__':main()
