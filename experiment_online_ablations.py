"""Run online ablations for Fig. 2(d), 2(e), and 2(f)."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from experiment_online import run

ALGORITHMS = ['CUCB-SC', 'CLCB-SC', 'CLCB-SC-LS', 'Epsilon-Greedy', 'LFU']
COLORS = {'CUCB-SC': '#ff7f0e', 'CLCB-SC': '#1f77b4', 'CLCB-SC-LS': '#2ca02c',
          'Epsilon-Greedy': '#d62728', 'LFU': '#9467bd'}


def load_inputs(folder, m, seeds):
    with np.load(folder / 'embedding_and_distance.npz') as data:
        p, D = data['probabilities'][:m].copy(), data['distances'][:m, :m].copy()
    p /= p.sum()
    costs = []
    for seed in range(seeds):
        with np.load(folder / f'seed_{seed:02d}_inputs.npz') as data:
            costs.append(data['costs'][:m].copy())
    return p, D, costs


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_case(p, D, costs, k, rounds, delta, alpha, label):
    all_rows, summary = [], []
    for name in ALGORITHMS:
        started = time.perf_counter()
        end_rows = []
        for seed, c in enumerate(costs):
            rows = run(p, c, D, k, rounds, seed, name, alpha, delta)
            all_rows.extend([dict(case=label, k=k, m=len(p), **row) for row in rows])
            end_rows.append(rows[-1])
        elapsed = time.perf_counter() - started
        summary.append({
            'case': label, 'k': k, 'm': len(p), 'algorithm': name,
            'final_mean_excess': float(np.mean([row['average_excess_cost'] for row in end_rows])),
            'final_sd_excess': float(np.std([row['average_excess_cost'] for row in end_rows], ddof=1)) if len(end_rows) > 1 else 0.0,
            'mean_cache_switches': float(np.mean([row['cache_switches'] for row in end_rows])),
            'mean_runtime_seconds': elapsed / len(costs),
        })
        print(f'{label}: {name} done', flush=True)
    return all_rows, summary


def plot_metric(rows, out, xfield, xvalues, metric, ylabel, title, filename):
    try:
        import matplotlib
    except ImportError:
        print('Plot skipped: install matplotlib to generate PNG files.', flush=True)
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if xfield == 'algorithm':
        values = [next(row[metric] for row in rows if row['algorithm'] == name) for name in ALGORITHMS]
        ax.bar(ALGORITHMS, values, color=[COLORS[name] for name in ALGORITHMS])
        ax.tick_params(axis='x', labelrotation=25)
        ax.set(ylabel=ylabel, title=title)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=200)
        plt.close(fig)
        return
    for name in ALGORITHMS:
        group = [row for row in rows if row['algorithm'] == name]
        means, stds = [], []
        for value in xvalues:
            vals = [row[metric] for row in group if row[xfield] == value]
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        x = np.asarray(xvalues)
        y, sd = np.asarray(means), np.asarray(stds)
        ax.plot(x, y, marker='o', label=name, color=COLORS[name])
        ax.fill_between(x, y - sd, y + sd, color=COLORS[name], alpha=.12)
    ax.set(xlabel=xfield, ylabel=ylabel, title=title)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / filename, dpi=200)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root / 'fig2a_results')
    parser.add_argument('--out', type=Path, default=root / 'fig2def_results')
    parser.add_argument('--rounds', type=int, default=5000)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--k-values', type=int, nargs='+', default=[2, 3, 5, 7, 10])
    parser.add_argument('--m-values', type=int, nargs='+', default=[10, 15, 20])
    parser.add_argument('--k-for-m', type=int, default=5)
    parser.add_argument('--delta', type=float, default=.05)
    parser.add_argument('--alpha', type=float, default=1.)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.input / 'embedding_and_distance.npz') as data:
        max_m = len(data['probabilities'])
    if any(m < 2 or m > max_m for m in args.m_values):
        parser.error(f'm-values must be in [2, {max_m}] for the available input data')
    if any(k < 1 or k >= max_m for k in args.k_values) or args.k_for_m >= min(args.m_values):
        parser.error('k must be positive and smaller than m for every experiment')

    d_rows, d_summary = [], []
    p20, D20, costs20 = load_inputs(args.input, max_m, args.seeds)
    for k in args.k_values:
        rows, summary = run_case(p20, D20, costs20, k, args.rounds, args.delta, args.alpha, f'k={k}')
        d_rows.extend(rows)
        d_summary.extend(summary)
    write_csv(args.out / 'fig2d_online.csv', d_rows)
    write_csv(args.out / 'fig2d_summary.csv', d_summary)
    plot_metric(d_summary, args.out, 'k', args.k_values, 'final_mean_excess',
                'Final average excess cost', 'Online ablation by cache size (m=20)', 'fig2d_by_k.png')

    e_rows, e_summary = [], []
    for m in args.m_values:
        p, D, costs = load_inputs(args.input, m, args.seeds)
        rows, summary = run_case(p, D, costs, args.k_for_m, args.rounds, args.delta, args.alpha, f'm={m}')
        e_rows.extend(rows)
        e_summary.extend(summary)
    write_csv(args.out / 'fig2e_online.csv', e_rows)
    write_csv(args.out / 'fig2e_summary.csv', e_summary)
    plot_metric(e_summary, args.out, 'm', args.m_values, 'final_mean_excess',
                'Final average excess cost', f'Online ablation by query count (k={args.k_for_m})', 'fig2e_by_m.png')

    f_summary = [row for row in d_summary if row['k'] == args.k_for_m]
    if not f_summary:
        p, D, costs = load_inputs(args.input, max_m, args.seeds)
        _, f_summary = run_case(p, D, costs, args.k_for_m, args.rounds, args.delta, args.alpha, f'f-k={args.k_for_m}')
    write_csv(args.out / 'fig2f_summary.csv', f_summary)
    plot_metric(f_summary, args.out, 'algorithm', ALGORITHMS, 'mean_cache_switches',
                'Mean cache switches', f'Cache switches (m=20, k={args.k_for_m})', 'fig2f_switches.png')
    plot_metric(f_summary, args.out, 'algorithm', ALGORITHMS, 'mean_runtime_seconds',
                'Runtime per seed (seconds)', f'Runtime (m=20, k={args.k_for_m})', 'fig2f_runtime.png')
    config = vars(args).copy()
    config.update({
        'input_data_note': 'For m<20, use the first m existing queries, distances, and costs, renormalizing uniform probabilities. The paper does not specify the query sets/probabilities for the m ablation.',
        'metric_note': 'Final average excess cost uses the exact optimum under each fixed input. Runtime is measured around each algorithm across seeds and divided by seed count; includes Python overhead, excludes input loading and plotting.',
        'limitations': 'Reference reconstruction from paper text; not guaranteed numerical replication of published values.',
    })
    (args.out / 'config.json').write_text(json.dumps(config, default=str, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Results saved to {args.out.resolve()}')


if __name__ == '__main__':
    main()

