"""
Evaluation identical to the original research code (h_result_analysis):
  * rating prediction: RMSE and MAE on all test interactions;
  * ranking: for each test user, the first held-out test interaction is the positive item (regardless of its
    rating) and is ranked against 99 items sampled uniformly from the items the user has not interacted with in
    the training, validation or test set (numpy seed 42, same sampling loop as the paper);
    rank = number of candidates scored strictly higher than the positive item;
    Recall@K = 1[rank < K], NDCG@K = 1[rank < K] / log2(rank + 2), averaged over test users.
"""
import json
import numpy as np
import pandas as pd
from .data import to_inputs


def rating_metrics(model, test):
    pred = model.predict(to_inputs(test), batch_size=2048, verbose=0).flatten()
    err = test['rating'].values - pred
    return {'RMSE': float(np.sqrt(np.mean(err ** 2))), 'MAE': float(np.mean(np.abs(err)))}, pred


def ranking_metrics(model, train, val, test, k_list=(10, 20), batch_user_size=100, random_seed=42):
    np.random.seed(random_seed)
    all_data = pd.concat([train, val, test])
    meta = all_data[['item_idx', 'store_idx', 'average_rating', 'rating_number', 'price']] \
        .drop_duplicates('item_idx').set_index('item_idx')
    num_items = all_data['item_idx'].nunique()
    users = test['user_idx'].unique()
    hits = {k: [] for k in k_list}; ndcg = {k: [] for k in k_list}
    for s in range(0, len(users), batch_user_size):
        U, I = [], []
        for u in users[s:s + batch_user_size]:
            pos = int(test[test['user_idx'] == u].iloc[0]['item_idx'])
            seen = set(all_data[all_data['user_idx'] == u]['item_idx'])
            neg = []
            while len(neg) < 99:
                j = np.random.randint(0, num_items)
                if j not in seen and j in meta.index:
                    neg.append(j)
            U.extend([u] * 100); I.extend([pos] + neg)
        m = meta.loc[I]
        X = [np.array(U, dtype=np.int64), np.array(I, dtype=np.int64), m['store_idx'].values.astype(np.int32),
             m['average_rating'].values, m['rating_number'].values, m['price'].values]
        P = model.predict(X, batch_size=2048, verbose=0).flatten().reshape(-1, 100)
        rank = np.sum(P > P[:, :1], axis=1)
        for k in k_list:
            h = (rank < k).astype(float)
            hits[k].extend(h); ndcg[k].extend(h / np.log2(rank + 2))
    out = {}
    for k in k_list:
        out[f'Recall@{k}'] = float(np.mean(hits[k])); out[f'NDCG@{k}'] = float(np.mean(ndcg[k]))
    out['n_test_users'] = int(len(users))
    return out


def evaluate(model, train, val, test, k_list=(10, 20)):
    rm, pred = rating_metrics(model, test)
    return {**rm, **ranking_metrics(model, train, val, test, k_list)}, pred


def save_predictions(path, test, pred):
    with open(path, 'w') as f:
        json.dump({'user_id': test['user_idx'].tolist(), 'item_id': test['item_idx'].tolist(),
                   'y_true': test['rating'].tolist(), 'y_pred': [float(x) for x in pred]}, f)
