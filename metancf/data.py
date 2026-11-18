"""
Amazon Fashion data loading, preprocessing and splitting.

The logic is identical to the original research code (h_data.py), which produced the reported results:
  * reviews: (user_id, parent_asin or asin, rating); users and items with fewer than five ratings in the raw
    data are removed in a single pass based on these raw counts (so some users/items keep fewer than five);
  * item attributes from the Amazon item metadata: store, price, average_rating, rating_number;
    missing values: store -> 'Unknown_Store', price -> median, average_rating -> mean, rating_number -> 0;
  * raw pipeline (paper's MetaNCF, NCF-Concat): numeric attributes kept as floats;
    binned pipeline (MetaNCF_bin): average_rating -> 10 equal-width bins, rating_number and price -> 10 quantile
    bins, all as integer ids;
  * split: shuffle with seed 42, then 80% train / 10% validation / 10% test (sklearn train_test_split, seed 42).
"""
import os, re, gzip, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

REVIEW_FILE = 'Amazon_Fashion.jsonl.gz'
META_FILE = 'meta_Amazon_Fashion.jsonl.gz'
INPUT_COLS = ['user_idx', 'item_idx', 'store_idx', 'average_rating', 'rating_number', 'price']


def load_reviews(path, min_count=5):
    data = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                user_id, item_id, rating = r.get('user_id'), r.get('parent_asin') or r.get('asin'), r.get('rating')
                if user_id and item_id and rating:
                    data.append({'user_id': str(user_id), 'item_id': str(item_id), 'rating': float(rating)})
            except Exception:
                continue
    ratings = pd.DataFrame(data)
    print(f'raw reviews: {len(ratings):,}')
    user_counts, item_counts = ratings['user_id'].value_counts(), ratings['item_id'].value_counts()
    ratings = ratings[ratings['user_id'].isin(user_counts[user_counts >= min_count].index)]
    ratings = ratings[ratings['item_id'].isin(item_counts[item_counts >= min_count].index)]
    user_ids, item_ids = sorted(ratings['user_id'].unique()), sorted(ratings['item_id'].unique())
    ratings['user_idx'] = ratings['user_id'].map({u: i for i, u in enumerate(user_ids)})
    ratings['item_idx'] = ratings['item_id'].map({it: i for i, it in enumerate(item_ids)})
    print(f'after filtering: {len(ratings):,} interactions, {len(user_ids):,} users, {len(item_ids):,} items')
    return ratings, len(user_ids), len(item_ids)


def load_metadata(path):
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                it = json.loads(line.strip())
                item_id = it.get('parent_asin') or it.get('asin')
                avg, num, price, store = it.get('average_rating'), it.get('rating_number'), it.get('price'), it.get('store')
                if item_id and avg and num:
                    price_val = None
                    if price:
                        try:
                            m = re.search(r'[\d,]+\.?\d*', str(price))
                            if m:
                                price_val = float(m.group().replace(',', ''))
                        except Exception:
                            pass
                    rows.append({'item_id': str(item_id), 'average_rating': float(avg), 'rating_number': int(num),
                                 'price': price_val, 'store': store})
            except Exception:
                continue
    df = pd.DataFrame(rows)
    print(f'item metadata records: {len(df):,}')
    return df


def build_features(ratings, meta, binned=False, num_bins=10):
    df = pd.merge(ratings, meta, on='item_id', how='left')
    df['price'] = df['price'].fillna(df['price'].median())
    df['average_rating'] = df['average_rating'].fillna(df['average_rating'].mean())
    df['rating_number'] = df['rating_number'].fillna(0)
    df['store'] = df['store'].fillna('Unknown_Store')
    if binned:
        df['average_rating'] = pd.cut(df['average_rating'], bins=num_bins, labels=False, include_lowest=True)
        df['rating_number'] = pd.qcut(df['rating_number'].rank(method='first'), q=num_bins, labels=False)
        df['price'] = pd.qcut(df['price'].rank(method='first'), q=num_bins, labels=False)
    enc = LabelEncoder()
    df['store_idx'] = enc.fit_transform(df['store'])
    out = df[['user_idx', 'item_idx', 'rating', 'average_rating', 'rating_number', 'price', 'store_idx']].copy()
    if binned:
        out[['average_rating', 'rating_number', 'price', 'store_idx']] = \
            out[['average_rating', 'rating_number', 'price', 'store_idx']].astype('int32')
    else:
        out[['average_rating', 'rating_number', 'price']] = out[['average_rating', 'rating_number', 'price']].astype('float32')
    return out, len(enc.classes_)


def split(df, seed=42):
    np.random.seed(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train, temp = train_test_split(df, test_size=0.2, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    return train, val, test


def load_dataset(data_dir, binned=False, num_bins=10, seed=42, cache_dir=None):
    """Returns dict(train, val, test, num_users, num_items, num_stores). Optionally caches the result."""
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f'amazon_fashion_{"binned%d" % num_bins if binned else "raw"}_seed{seed}.pkl')
        if os.path.exists(cache):
            with open(cache, 'rb') as f:
                d = pickle.load(f)
            print(f'loaded cached dataset {cache}')
            return d
    ratings, num_users, num_items = load_reviews(os.path.join(data_dir, REVIEW_FILE))
    meta = load_metadata(os.path.join(data_dir, META_FILE))
    df, num_stores = build_features(ratings, meta, binned=binned, num_bins=num_bins)
    train, val, test = split(df, seed)
    print(f'split: train={len(train):,} val={len(val):,} test={len(test):,} | stores={num_stores:,}')
    d = dict(train=train, val=val, test=test, num_users=num_users, num_items=num_items, num_stores=num_stores)
    if cache:
        with open(cache, 'wb') as f:
            pickle.dump(d, f)
    return d


def to_inputs(df):
    return [df[c].values for c in INPUT_COLS]
