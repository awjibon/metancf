"""
NCF to MetaNCF ablation on Amazon Fashion (Table 6 of the paper): NCF (MLP-only), NCF, NCF-Concat and MetaNCF
(concat fusion), each trained and evaluated in four runs; reports mean and standard deviation over runs.

L2 regularization per run, as used for the reported results:
  NCF, NCF-Concat, MetaNCF: runs 1 and 4 -> 1e-3, runs 2 and 3 -> 1e-2;   NCF (MLP-only): 1e-2 in all runs.
Optional: --models ... metancf_bin to add MetaNCF with binned attributes (same L2 schedule as MetaNCF).

Examples
  python run_ablation.py --smoke              # quick installation check (tiny subset, 1 run, 2 epochs)
  python run_ablation.py                      # full ablation, outputs/ablation/
  python run_ablation.py --models ncf metancf --runs 1 2
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import train_test
from metancf.data import load_dataset
from metancf.models import BINNED_MODELS
from metancf.utils import setup_gpu, Tee, smoke_subset

L2_SCHEDULE = {1: 1e-3, 2: 1e-2, 3: 1e-2, 4: 1e-3}
L2_OVERRIDE = {'ncf_mlp': 1e-2}
DEFAULT_MODELS = ['ncf_mlp', 'ncf', 'ncf_concat', 'metancf']
METRICS = ['RMSE', 'MAE', 'Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=DEFAULT_MODELS, choices=['ncf_mlp', 'ncf_gmf', 'ncf', 'ncf_concat', 'metancf', 'metancf_bin'])
    ap.add_argument('--runs', nargs='+', type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--out_dir', default='outputs/ablation')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--gpu_mem_mb', type=int, default=None)
    ap.add_argument('--no_seed', action='store_true', help='do not fix the TensorFlow seed (seed = 100 + run id otherwise)')
    a = ap.parse_args()
    out = a.out_dir + ('_smoke' if a.smoke else '')
    os.makedirs(out, exist_ok=True)
    sys.stdout = sys.stderr = Tee(os.path.join(out, 'log.txt'))
    print('=' * 70); print('ablation', time.strftime('%Y-%m-%d %H:%M:%S'), vars(a))
    setup_gpu(a.gpu_mem_mb)
    runs = a.runs[:1] if a.smoke else a.runs
    datasets = {}
    for binned in sorted({m in BINNED_MODELS for m in a.models}):
        d = load_dataset(a.data_dir, binned=binned, cache_dir=os.path.join('outputs', 'cache'))
        datasets[binned] = smoke_subset(d) if a.smoke else d

    rows = []
    for run in runs:
        for m in a.models:
            res_dir = os.path.join(out, f'run{run}', m)
            mfile = os.path.join(res_dir, 'metrics.json')
            if os.path.exists(mfile):
                print(f'[run {run}] {m}: already done, skipping')
                with open(mfile) as f: met = json.load(f)['metrics']
            else:
                l2 = L2_OVERRIDE.get(m, L2_SCHEDULE[run])
                print('-' * 70); print(f'[run {run}] {m} | L2={l2:g}')
                args = train_test.get_args(['--model', m, '--l2', str(l2), '--out_dir', res_dir,
                                            '--seed', str(-1 if a.no_seed else 100 + run)] + (['--smoke'] if a.smoke else []))
                met = train_test.run(args, data=datasets[m in BINNED_MODELS])
            rows.append({'run': run, 'model': m, **{k: met[k] for k in METRICS}})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, 'all_runs.csv'), index=False)
    summ = df.groupby('model', sort=False)[METRICS].agg(['mean', 'std'])
    summ.to_csv(os.path.join(out, 'summary.csv'))
    print('=' * 70); print(f'mean +- SD over runs {runs}')
    for m in a.models:
        if m in summ.index:
            print(f'{m:12s} ' + '  '.join(f"{k} {summ.loc[m, (k, 'mean')]:.4f}+-{(summ.loc[m, (k, 'std')] if len(runs) > 1 else 0):.4f}" for k in METRICS))
    print('DONE', time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(); sys.stdout.flush(); raise
