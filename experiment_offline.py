"""Offline semantic caching experiment corresponding to Fig. 2(b)."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from experiment_known_parameters import loss, reverse_greedy, brute_force


def make_log(p, c, n, seed, observation_probability, feedback_std):
    """Synthetic logging: empty cache with probability nu; otherwise full cache.
    This explicitly chosen logging policy gives cost feedback probability nu for every query.
    Cache policies are stored as empty/full codes instead of repeating all indices.
    """
    child_seeds = np.random.SeedSequence([seed, 20261026]).spawn(3)
    arrival_rng, policy_rng, noise_rng = [np.random.default_rng(s) for s in child_seeds]
    q = arrival_rng.choice(len(p), size=n, p=p)
    observed = policy_rng.random(n) < observation_probability
    realized = c[q] + noise_rng.normal(0, feedback_std, n)
    feedback = np.where(observed, realized, np.nan)
    return q, feedback


def estimate_from_log(q, feedback, m, delta):
    n = len(q)
    available = ~np.isnan(feedback)  # Observed zero costs are valid; zero does not mean missing.
    counts = np.bincount(q, minlength=m)
    cost_counts = np.bincount(q[available], minlength=m)
    sums = np.bincount(q[available], weights=feedback[available], minlength=m)
    phat = counts / n
    mean = np.divide(sums, cost_counts, out=np.zeros(m), where=cost_counts > 0)
    radius = np.full(m, np.inf)
    known = cost_counts > 0
    radius[known] = np.sqrt(np.log(6*m*n/delta)/(2*cost_counts[known]))
    # Paper formula, without silent clipping. For unobserved queries use [0,1].
    ucb = np.where(known, mean + radius, 1.0)
    lcb = np.where(known, mean - radius, 0.0)
    empirical = np.where(known, mean, 1.0)
    return phat, mean, cost_counts, ucb, lcb, empirical


def epsilon_greedy(phat, empirical, D, k, rng, epsilon=.2):
    M = list(range(len(phat)))
    while len(M) > k:
        if rng.random() < epsilon:
            removed = int(rng.choice(M))
        else:
            removed = min(M, key=lambda q: (loss([i for i in M if i != q], phat, empirical, D), q))
        M.remove(removed)
    return M


def select_caches(q, feedback, D, k, delta, rng, epsilon):
    # No access to true p or c in this function.
    phat, mean, nc, ucb, lcb, empirical = estimate_from_log(q, feedback, len(D), delta)
    selected = {
        'CUCB-SC': reverse_greedy(phat, ucb, D, k),
        'CLCB-SC': reverse_greedy(phat, lcb, D, k),
        'Epsilon-Greedy': epsilon_greedy(phat, empirical, D, k, rng, epsilon),
        'LFU': np.argsort(-phat, kind='stable')[:k].tolist(),
    }
    return selected, phat, mean, nc, ucb, lcb


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=root/'fig2a_results')
    ap.add_argument('--out', type=Path, default=root/'fig2b_results')
    ap.add_argument('--lengths', type=int, nargs='+', default=list(range(300, 601, 25)))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--delta', type=float, default=.05)
    ap.add_argument('--observation-probability', type=float, default=.5)
    ap.add_argument('--feedback-std', type=float, default=0.0)
    ap.add_argument('--epsilon', type=float, default=.2)
    ap.add_argument('--alpha', type=float, default=1.0)
    args = ap.parse_args()
    if args.seeds < 1 or min(args.lengths) < 1 or not 0 < args.delta < 1 or not 0 < args.observation_probability <= 1 or args.feedback_std < 0 or not 0 <= args.epsilon <= 1 or not np.isfinite(args.alpha) or args.alpha < 1:
        ap.error('Invalid experimental settings.')
    args.lengths = sorted(set(args.lengths))
    with np.load(args.input/'embedding_and_distance.npz', allow_pickle=False) as a:
        p, D = a['probabilities'].copy(), a['distances'].copy()
    queries = json.loads((args.input/'queries.json').read_text(encoding='utf-8'))
    if len(p) != 20 or D.shape != (20,20) or len(queries) != 20:
        ap.error('Expected m=20 data from the known-parameter experiment.')
    if not np.isfinite(D).all() or np.any(D < 0) or not np.allclose(D,D.T) or not np.allclose(np.diag(D),0) or not np.allclose(p,1/20):
        ap.error('Invalid distance matrix or expected uniform arrival probabilities.')
    # Validate all input cost files before writing results.
    all_costs=[]
    for seed in range(args.seeds):
        with np.load(args.input/f'seed_{seed:02d}_inputs.npz',allow_pickle=False) as a:
            c=a['costs'].copy()
        if c.shape != (20,) or not np.isfinite(c).all() or np.any(c<0) or np.any(c>1):
            ap.error(f'Invalid costs for seed {seed}')
        all_costs.append(c)
    args.out.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy()
    config['input']=str(args.input.resolve()); config['out']=str(args.out.resolve())
    config['m']=20; config['k']=5
    config['logging_policy']='For each request use empty cache with probability nu, otherwise full query cache; empty means invoke, full means reuse. Logging caches need not have target capacity 5.'
    config['cost_convention']='Reuse fixed costs generated by token count + Gaussian std .05 then min-max in Fig2a. Feedback std defaults to 0 to avoid adding a second unconfirmed noise process. Optional nonzero feedback_std explicitly changes the observation model.'
    config['bounds']='Raw mean +/- sqrt(log(6mn/delta)/(2Nc)); no clipping. Nc=0: UCB=1, LCB=0, empirical=1. Negative LCB is retained for selection only.'
    config['metric']='Always save excess_loss=loss-optimum and suboptimality=loss-alpha*optimum. Default alpha=1 is exact-optimum gap, not the uncomputed curvature factor in Theorem 1.'
    config['limitations']='Original queries, logging policy, tokenizer, checkpoint, delta, zero-count rule and plotted alpha not specified sufficiently. This is an explicit reference experiment, not exact numerical reproduction. LLM cache-answer filling is simulated, not called or charged in static Eq(2).'
    config['packages']={x:importlib.metadata.version(x) for x in ['numpy','matplotlib']}
    (args.out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.out/'queries.json').write_text(json.dumps(queries,indent=2),encoding='utf-8')
    rows=[]; estimates=[]
    for seed,c in enumerate(all_costs):
        best_cache,optimum=brute_force(p,c,D,5)
        q,feedback=make_log(p,c,max(args.lengths),seed,args.observation_probability,args.feedback_std)
        # Same log prefix for every algorithm and every shorter n.
        np.savez(args.out/f'seed_{seed:02d}_log.npz',query_indices=q,cost_feedback=feedback,empty_cache=np.isfinite(feedback),true_costs=c,optimal_cache=best_cache,optimal_loss=optimum)
        for n in args.lengths:
            rng=np.random.default_rng(np.random.SeedSequence([seed,n,12345]))
            caches,phat,mean,nc,ucb,lcb=select_caches(q[:n],feedback[:n],D,5,args.delta,rng,args.epsilon)
            for i in range(20):
                estimates.append(dict(seed=seed,n=n,query_index=i,p_hat=phat[i],c_hat=mean[i] if nc[i] else '',cost_observations=int(nc[i]),ucb=ucb[i],lcb=lcb[i]))
            for name,M in caches.items():
                value=loss(M,p,c,D)
                if value < optimum-1e-10: raise AssertionError('Loss below exact optimum.')
                rows.append(dict(seed=seed,n=n,algorithm=name,loss=value,optimal_loss=optimum,excess_loss=value-optimum,alpha=args.alpha,suboptimality=value-args.alpha*optimum,unobserved_queries=int((nc==0).sum()),cache_indices=json.dumps(M)))
        print(f'Completed seed {seed+1}/{args.seeds}',flush=True)
    write_csv(args.out/'all_runs.csv',rows); write_csv(args.out/'parameter_estimates.csv',estimates)
    summary=[]
    for name in ['CLCB-SC','CUCB-SC','Epsilon-Greedy','LFU']:
        for n in args.lengths:
            g=[r for r in rows if r['algorithm']==name and r['n']==n]
            v=np.array([r['suboptimality'] for r in g]); e=np.array([r['excess_loss'] for r in g])
            summary.append(dict(algorithm=name,n=n,mean_gap=float(v.mean()),std_gap=float(v.std(ddof=1)) if len(v)>1 else 0.,mean_excess_loss=float(e.mean())))
    write_csv(args.out/'summary.csv',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    for name,color,marker in [('CLCB-SC','#1f77b4','o'),('CUCB-SC','#ff7f0e','v'),('Epsilon-Greedy','#d62728','s'),('LFU','#9467bd','^')]:
        g=[r for r in summary if r['algorithm']==name]
        x=np.array([r['n'] for r in g]); y=np.array([r['mean_gap'] for r in g]); sd=np.array([r['std_gap'] for r in g])
        ax.plot(x,y,color=color,marker=marker,label=name,markersize=4)
        ax.fill_between(x,y-sd,y+sd,color=color,alpha=.15)
    ax.set(xlabel='Stream Length (n)',ylabel='Final gap to exact optimum' if args.alpha==1 else f'Final suboptimality (alpha={args.alpha:g})',title='Offline Semantic Caching: m=20, k=5')
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(args.out/'fig2b_offline.png',dpi=200); plt.close(fig)
    print('Results saved to',args.out.resolve())


if __name__=='__main__':
    main()
