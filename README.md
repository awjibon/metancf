# MetaNCF-v1

Code for **MetaNCF**: personalized attribute attention for neural collaborative filtering, evaluated on the
Amazon Fashion dataset. This folder contains the models and the scripts for the NCF-to-MetaNCF ablation
(Table 6 of the paper) and for training and testing any single model, including MetaNCF with binned attributes.

## Contents

```
MetaNCF-v1/
├── metancf/
│   ├── data.py        # loading, filtering, attribute preprocessing (raw or binned), 80/10/10 split
│   ├── models.py      # NCF, NCF (MLP-only / GMF-only), NCF-Concat, MetaNCF, MetaNCF_bin
│   ├── train.py       # training loop (Adam, MSE, early stopping on validation MSE)
│   ├── evaluate.py    # RMSE/MAE and ranking (1 positive + 99 sampled negatives, Recall/NDCG@10,20)
│   └── utils.py
├── run_ablation.py    # NCF (MLP-only), NCF, NCF-Concat, MetaNCF: 4 runs, mean ± SD
├── train_test.py      # train and/or test one model
├── run_smoke_test.bat # Windows: quick installation check
├── run_ablation.bat   # Windows: full ablation
├── requirements.txt
└── data/              # put the two dataset files here
```

## 1. Setup

Tested with Python 3.8.10 and TensorFlow 2.9.0 on an NVIDIA GeForce RTX 3090 (Windows 11).

```bash
conda create -n metancf python=3.8
conda activate metancf
pip install -r requirements.txt
```

Newer TensorFlow versions (Keras 3, TensorFlow 2.16 and later) change the saving API and are not supported.

## 2. Data

Download the **Amazon Fashion** files of the *Amazon Reviews 2023* dataset (McAuley Lab,
https://amazon-reviews-2023.github.io/) and place them in `data/`:

```
data/Amazon_Fashion.jsonl.gz        # user reviews (ratings)
data/meta_Amazon_Fashion.jsonl.gz   # item metadata (store, price, average_rating, rating_number)
```

Use `--data_dir <folder>` if the files are elsewhere. Preprocessing (identical for all models):

* interactions: `(user_id, parent_asin, rating)`; users and items with fewer than five ratings in the raw data are
  removed in a single pass based on the raw counts, giving 14,494 users, 27,327 items and 53,027 interactions;
* item attributes: store, price, average rating and rating number from the item metadata; missing values:
  store → `Unknown_Store`, price → median, average rating → mean, rating number → 0;
* **raw** attributes (NCF-Concat, MetaNCF): numeric attributes are used as floats;
  **binned** attributes (MetaNCF_bin): average rating → 10 equal-width bins, rating number and price → 10 quantile
  bins, each embedded like the store;
* split: shuffled with seed 42, then 80% training / 10% validation / 10% test (42,421 / 5,303 / 5,303).

The first run caches the preprocessed data in `outputs/cache/`, which makes later runs start faster.

## 3. Quick check

```bash
python run_ablation.py --smoke
python train_test.py --model metancf_bin --smoke
```

The smoke test trains each model for two epochs on a small subset; it only checks that everything runs.
On Windows, `run_smoke_test.bat` does the same (it activates the conda environment `tf2` and uses `..\code`
as the data folder; edit these two names if needed).

## 4. NCF-to-MetaNCF ablation (Table 6)

```bash
python run_ablation.py                    # 4 models x 4 runs, outputs in outputs/ablation/
python run_ablation.py --models ncf metancf --runs 1 2      # a subset
python run_ablation.py --models ncf_mlp ncf ncf_concat metancf metancf_bin   # add the binned MetaNCF
```

| Model (`--models`) | Description |
|---|---|
| `ncf_mlp` | NCF with the MLP path only |
| `ncf` | NCF (NeuMF: GMF + MLP), no attributes |
| `ncf_concat` | NCF-Concat: item attributes concatenated to the MLP input (static fusion) |
| `metancf` | MetaNCF: personalized attribute attention, concat fusion, raw attributes |
| `ncf_gmf`, `metancf_bin` | optional: NCF with the GMF path only; MetaNCF with binned attributes |

Hyperparameters (as used for the reported results): Adam with learning rate 1e-3, batch size 256, MSE loss on
the ratings, at most 100 epochs with early stopping on the validation MSE (patience 10, best weights restored).
MetaNCF and NCF-Concat use 16-dimensional embeddings and an MLP with 64-32-32-16 units; MetaNCF uses four
attention heads. NCF uses the original NeuMF sizes (8-dimensional GMF and 32-dimensional MLP embeddings, MLP
64-32-16-8). The L2 regularization differs between runs:

| Run | NCF, NCF-Concat, MetaNCF | NCF (MLP-only) |
|---|---|---|
| 1 | 1e-3 | 1e-2 |
| 2 | 1e-2 | 1e-2 |
| 3 | 1e-2 | 1e-2 |
| 4 | 1e-3 | 1e-2 |

Outputs: `outputs/ablation/run<k>/<model>/` (weights, config, training history, test predictions, metrics),
`outputs/ablation/all_runs.csv`, `outputs/ablation/summary.csv` (mean and SD) and `log.txt`.
Completed (run, model) pairs are skipped, so an interrupted ablation can simply be restarted.

Reported results (mean ± SD over four runs):

| Model | RMSE | MAE | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| NCF (MLP-only) | 1.1441 ± 0.0035 | 0.8712 ± 0.0181 | 0.0898 ± 0.0118 | 0.1592 ± 0.0135 | 0.0460 ± 0.0071 | 0.0634 ± 0.0076 |
| NCF | 1.1584 ± 0.0182 | 0.8997 ± 0.0246 | 0.1092 ± 0.0391 | 0.1709 ± 0.0332 | 0.0560 ± 0.0226 | 0.0716 ± 0.0212 |
| NCF-Concat | 1.1769 ± 0.0242 | 0.9251 ± 0.0320 | 0.0983 ± 0.0082 | 0.1891 ± 0.0115 | 0.0481 ± 0.0033 | 0.0707 ± 0.0040 |
| MetaNCF | 1.1410 ± 0.0089 | 0.8732 ± 0.0195 | 0.1235 ± 0.0334 | 0.2126 ± 0.0362 | 0.0641 ± 0.0219 | 0.0863 ± 0.0217 |

Training on a GPU is not bit-wise deterministic, and the runs reported in the paper were not seeded (except the
MLP-only runs), so a rerun gives values within the run-to-run variation shown above rather than identical numbers.
This code seeds TensorFlow with `100 + run id` by default (`--no_seed` disables it).

## 5. Training and testing a single model

```bash
python train_test.py --model metancf --l2 1e-3                      # MetaNCF, outputs/metancf/
python train_test.py --model metancf_bin --l2 1e-3                  # MetaNCF with binned attributes
python train_test.py --model ncf --l2 1e-3 --out_dir outputs/ncf_l2_1e-3
python train_test.py --model metancf --out_dir outputs/metancf --test_only   # evaluate saved weights
```

Main options (`python train_test.py -h` lists all):

| Option | Default | Meaning |
|---|---|---|
| `--model` | (required) | `ncf_mlp`, `ncf_gmf`, `ncf`, `ncf_concat`, `metancf`, `metancf_bin` |
| `--l2` | 1e-3 | L2 regularization |
| `--fusion` | concat | MetaNCF fusion: `concat`, `additive`, `residual`, `gated` |
| `--latent_dim`, `--heads` | 16, 4 | MetaNCF embedding size and attention heads |
| `--tau`, `--num_bins` | 0.01, 10 | attention temperature and bins per attribute (`metancf_bin` only) |
| `--seed` | 1 | TensorFlow seed (`-1`: none) |
| `--test_only` | off | load `<out_dir>/weights` and only evaluate |
| `--data_dir`, `--out_dir` | `data`, `outputs/<model>` | input and output folders |

Each run writes `weights.*`, `config.json`, `history.json`, `test_predictions.json` and `metrics.json`
(RMSE, MAE, Recall@10/20, NDCG@10/20) to the output folder.

## 6. Evaluation protocol

* Rating prediction: RMSE and MAE over all test interactions.
* Ranking: for each test user, the first held-out test interaction is the positive item, regardless of its rating,
  and is ranked against 99 items sampled uniformly from the items the user has not interacted with in the
  training, validation or test set (numpy seed 42, so all models see the same candidates). The rank of the
  positive item is the number of candidates with a strictly higher predicted score;
  Recall@K = 1[rank < K] and NDCG@K = 1[rank < K] / log2(rank + 2), averaged over the 4,326 test users.
