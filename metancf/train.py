"""Training loop identical to the original research code (h_train_test.run_experiment)."""
import os, json
import numpy as np
import tensorflow as tf
from .data import to_inputs


def train_model(model, train, val, learning_rate=1e-3, patience=10, max_epochs=100, batch_size=256, verbose=2):
    """Adam + MSE on the ratings, early stopping on validation MSE (best weights restored)."""
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae', 'mse'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=patience, restore_best_weights=True, verbose=1)
    hist = model.fit(to_inputs(train), train['rating'].values, validation_data=(to_inputs(val), val['rating'].values),
                     epochs=max_epochs, batch_size=batch_size, callbacks=[es], verbose=verbose)
    return {k: [float(x) for x in v] for k, v in hist.history.items()}


def build_weights(model, df):
    """Runs the model once so that its weights exist (needed before load_weights)."""
    model([np.asarray(x[:2]) for x in to_inputs(df)])


def save_model(model, out_dir, config, history=None):
    os.makedirs(out_dir, exist_ok=True)
    model.save_weights(os.path.join(out_dir, 'weights'))
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=1)
    if history is not None:
        with open(os.path.join(out_dir, 'history.json'), 'w') as f:
            json.dump(history, f)


def load_weights(model, out_dir, df):
    build_weights(model, df)
    model.load_weights(os.path.join(out_dir, 'weights')).expect_partial()
