"""
Train and/or test a single model on Amazon Fashion.

Examples
  python train_test.py --model metancf --l2 1e-3                     # MetaNCF (raw attributes, concat fusion)
  python train_test.py --model metancf_bin --l2 1e-3                 # MetaNCF with binned attributes
  python train_test.py --model ncf --l2 1e-3 --out_dir outputs/ncf
  python train_test.py --model metancf --out_dir outputs/metancf --test_only   # evaluate saved weights
Models: ncf_mlp, ncf_gmf, ncf, ncf_concat, metancf, metancf_bin (see metancf/models.py).
"""
import os, sys, json, time, argparse
from metancf.data import load_dataset
from metancf.models import build_model, MODEL_NAMES, BINNED_MODELS
from metancf.train import train_model, save_model, load_weights
from metancf.evaluate import evaluate, save_predictions
from metancf.utils import setup_gpu, set_seed, clear, Tee, smoke_subset


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=MODEL_NAMES)
    ap.add_argument('--data_dir', default='data', help='folder with Amazon_Fashion.jsonl.gz and meta_Amazon_Fashion.jsonl.gz')
    ap.add_argument('--out_dir', default=None, help='default: outputs/<model>')
    ap.add_argument('--l2', type=float, default=1e-3, help='L2 regularization of embeddings and MLP layers')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--max_epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--fusion', default='concat', choices=['concat', 'additive', 'residual', 'gated'], help='MetaNCF fusion')
    ap.add_argument('--latent_dim', type=int, default=16, help='MetaNCF embedding size')
    ap.add_argument('--heads', type=int, default=4, help='MetaNCF attention heads')
    ap.add_argument('--tau', type=float, default=0.01, help='attention temperature (metancf_bin only)')
    ap.add_argument('--num_bins', type=int, default=10, help='bins per numeric attribute (metancf_bin only)')
    ap.add_argument('--seed', type=int, default=1, help='TensorFlow/numpy seed for training (-1: no seed)')
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--test_only', action='store_true', help='load <out_dir>/weights and evaluate only')
    ap.add_argument('--smoke', action='store_true', help='tiny subset and 2 epochs, to check the installation')
    ap.add_argument('--gpu_mem_mb', type=int, default=None, help='cap GPU memory (MB); default: memory growth')
    return ap.parse_args(argv)


def run(args, data=None):
    out_dir = args.out_dir or os.path.join('outputs', args.model + ('_smoke' if args.smoke else ''))
    os.makedirs(out_dir, exist_ok=True)
    binned = args.model in BINNED_MODELS
    if data is None:
        data = load_dataset(args.data_dir, binned=binned, num_bins=args.num_bins, seed=args.split_seed,
                            cache_dir=os.path.join('outputs', 'cache'))
        if args.smoke:
            data = smoke_subset(data)
    tr, va, te = data['train'], data['val'], data['test']
    config = {k: getattr(args, k) for k in ['model', 'l2', 'lr', 'batch_size', 'max_epochs', 'patience', 'fusion',
                                            'latent_dim', 'heads', 'tau', 'num_bins', 'seed', 'split_seed']}
    config.update(binned=binned, num_users=data['num_users'], num_items=data['num_items'], num_stores=data['num_stores'])
    clear(); set_seed(None if args.seed < 0 else args.seed)
    model = build_model(args.model, data['num_users'], data['num_items'], data['num_stores'], args.l2,
                        fusion=args.fusion, latent_dim=args.latent_dim, num_heads=args.heads, tau=args.tau,
                        num_bins=args.num_bins)
    t0 = time.time()
    if args.test_only:
        load_weights(model, out_dir, tr)
        print(f'loaded weights from {out_dir}')
    else:
        hist = train_model(model, tr, va, learning_rate=args.lr, patience=args.patience,
                           max_epochs=2 if args.smoke else args.max_epochs, batch_size=args.batch_size)
        config.update(epochs=len(hist['loss']), train_minutes=round((time.time() - t0) / 60, 2),
                      n_params=int(model.count_params()))
        save_model(model, out_dir, config, hist)
    metrics, pred = evaluate(model, tr, va, te)
    save_predictions(os.path.join(out_dir, 'test_predictions.json'), te, pred)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({'config': config, 'metrics': metrics}, f, indent=1)
    print(f'[{args.model}] ' + ' '.join(f'{k} {v:.4f}' for k, v in metrics.items() if k != 'n_test_users'))
    del model; clear()
    return metrics


if __name__ == '__main__':
    a = get_args()
    os.makedirs('outputs', exist_ok=True)
    sys.stdout = sys.stderr = Tee(os.path.join('outputs', 'train_test_log.txt'))
    print('=' * 70); print('train_test', time.strftime('%Y-%m-%d %H:%M:%S'), vars(a))
    setup_gpu(a.gpu_mem_mb)
    try:
        run(a)
    except Exception:
        import traceback; traceback.print_exc(); sys.stdout.flush(); raise
