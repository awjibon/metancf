import gc, sys
import tensorflow as tf


def setup_gpu(memory_limit_mb=None):
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            if memory_limit_mb:
                tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)])
            else:
                tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:
            print('GPU setup:', e)
    print('GPUs:', [g.name for g in gpus] or 'none (CPU)')


def set_seed(seed):
    if seed is not None:
        tf.keras.utils.set_random_seed(int(seed))


def clear():
    tf.keras.backend.clear_session(); gc.collect()


class Tee:
    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8'); self.out = sys.stdout
    def write(self, s):
        self.out.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.out.flush(); self.f.flush()


def smoke_subset(d, n_train=5000, n_val=1000, n_users=200):
    d = dict(d)
    d['train'] = d['train'].sample(n=min(n_train, len(d['train'])), random_state=0)
    d['val'] = d['val'].sample(n=min(n_val, len(d['val'])), random_state=0)
    users = d['test']['user_idx'].unique()[:n_users]
    d['test'] = d['test'][d['test']['user_idx'].isin(users)]
    return d
