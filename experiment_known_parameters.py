"""Reproduce the experiment TYPE in Fig. 2(a); see KNOWN_PARAMETERS_README.md."""
import argparse
import csv
import importlib.metadata
import itertools
import json
from pathlib import Path
import numpy as np


def minmax(x):
    x = np.asarray(x, dtype=float)
    span = float(np.ptp(x))
    if span == 0:
        raise ValueError('Cannot min-max scale identical values.')
    return (x - x.min()) / span


def loss(M, p, c, D):
    nearest = D[:, list(M)].min(axis=1) if len(M) else np.full(len(p), np.inf)
    return float(p @ np.minimum(c, nearest))


def reverse_greedy(p, c, D, k):
    M = list(range(len(p)))
    while len(M) > k:
        removed = min(M, key=lambda q: (loss([j for j in M if j != q], p, c, D), q))
        M.remove(removed)
    return M


def brute_force(p, c, D, k):
    # Batched exact enumeration; no approximation or greedy substitution.
    combinations = itertools.combinations(range(len(p)), k)
    best_loss, best_cache = float('inf'), None
    while True:
        batch = list(itertools.islice(combinations, 2048))
        if not batch:
            break
        indices = np.array(batch, dtype=int)
        nearest = D[:, indices].min(axis=2)  # query x candidate cache
        values = (p[:, None] * np.minimum(c[:, None], nearest)).sum(axis=0)
        j = int(np.argmin(values))
        if float(values[j]) < best_loss:
            best_loss, best_cache = float(values[j]), list(batch[j])
    return best_cache, best_loss


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--queries', type=Path, default=root/'queries20.txt')
    ap.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    ap.add_argument('--revision', default=None)
    ap.add_argument('--k', type=int, nargs='+', default=[5, 7, 10])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--lfu-stream-length', type=int, default=10000)
    ap.add_argument('--out', type=Path, default=root/'fig2a_results')
    args = ap.parse_args()
    queries = [q.strip() for q in args.queries.read_text(encoding='utf-8-sig').splitlines() if q.strip()]
    m = len(queries)
    if m != 20 or len(set(queries)) != m:
        ap.error('This experiment requires 20 distinct queries.')
    if args.seeds < 1 or args.lfu_stream_length < 1 or any(k < 1 or k > m for k in args.k):
        ap.error('Invalid seeds, stream length or cache sizes.')
    from sentence_transformers import SentenceTransformer
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.out.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model, revision=args.revision, device='cpu')
    E = model.encode(queries, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)
    if E.shape != (m, 384) or not np.isfinite(E).all():
        raise ValueError(f'Expected finite (20,384) embeddings, got {E.shape}')
    tokens = np.array([len(model.tokenizer.encode(q, add_special_tokens=False, truncation=False)) for q in queries])
    raw_D = np.linalg.norm(E[:, None, :].astype(float)-E[None, :, :].astype(float), axis=-1)
    np.fill_diagonal(raw_D, 0)
    D = minmax(raw_D)
    p = np.full(m, 1/m)
    np.savez(args.out/'embedding_and_distance.npz', embeddings=E, distances=D, raw_distances=raw_D, token_counts=tokens, probabilities=p)
    (args.out/'queries.json').write_text(json.dumps(queries, indent=2), encoding='utf-8')
    config = vars(args).copy()
    for key in ('queries','out'): config[key] = str(config[key])
    config['packages'] = {x:importlib.metadata.version(x) for x in ['sentence-transformers','numpy','matplotlib','torch','transformers']}
    config['model_structure'] = str(model)
    config['cost_rule'] = 'One noise draw per query per seed: minmax(token_count + Normal(0,0.05^2)). Fixed across k and algorithms within seed; no second noise.'
    config['lfu_rule'] = 'Most frequent queries in a uniform sampled stream; ties by query index. Stream length is an implementation choice.'
    config['limitations'] = 'Self-created query set; tokenizer, checkpoint, noise sampling scope and LFU stream length not specified in paper. Min-max costs include zero. No guarantee of matching paper numbers.'
    (args.out/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    records = []
    for seed in range(args.seeds):
        cost_rng, arrival_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
        noise = cost_rng.normal(0, .05, m)
        c = minmax(tokens + noise)
        arrivals = arrival_rng.choice(m, size=args.lfu_stream_length, p=p)
        counts = np.bincount(arrivals, minlength=m)
        ranking = np.argsort(-counts, kind='stable')
        np.savez(args.out/f'seed_{seed:02d}_inputs.npz', noise=noise, costs=c, arrivals=arrivals, frequencies=counts)
        for k in args.k:
            M_g = reverse_greedy(p,c,D,k)
            M_b, best = brute_force(p,c,D,k)
            M_l = ranking[:k].tolist()
            for name,M in [('Brute Force',M_b),('Reverse Greedy',M_g),('LFU',M_l)]:
                value = loss(M,p,c,D)
                if value < best-1e-10:
                    raise AssertionError('Method beats exact optimum: implementation error.')
                records.append({'seed':seed,'k':k,'algorithm':name,'loss':value,'gap_to_optimum':value-best,'cache_indices':json.dumps(M)})
            print(f'Seed {seed+1}/{args.seeds}, k={k}: optimum={best:.6f}, greedy={loss(M_g,p,c,D):.6f}', flush=True)
    with (args.out/'all_runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary=[]
    for k in args.k:
        for name in ['Brute Force','Reverse Greedy','LFU']:
            values=np.array([r['loss'] for r in records if r['k']==k and r['algorithm']==name])
            summary.append({'k':k,'algorithm':name,'mean_loss':float(values.mean()),'std_loss':float(values.std(ddof=1)) if len(values)>1 else 0.})
    with (args.out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    fig,ax=plt.subplots(figsize=(7,4.5)); x=np.arange(len(args.k)); width=.24
    for j,(name,color,hatch) in enumerate([('Brute Force','#1f77b4','/'),('Reverse Greedy','#ff7f0e','\\'),('LFU','#9467bd','x')]):
        rows=[next(r for r in summary if r['k']==k and r['algorithm']==name) for k in args.k]
        ax.bar(x+(j-1)*width,[r['mean_loss'] for r in rows],width,yerr=[r['std_loss'] for r in rows],capsize=3,label=name,color=color,edgecolor='black',hatch=hatch)
    ax.set_xticks(x,args.k); ax.set(xlabel='Cache Size (k)',ylabel='Expected Loss',title='Known Parameters: Reverse Greedy vs Exact Optimum')
    ax.legend(); ax.grid(axis='y',alpha=.25); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(args.out/'fig2a_known_parameters.png',dpi=200); plt.close(fig)
    print('Results saved to',args.out.resolve())


if __name__ == '__main__':
    main()
