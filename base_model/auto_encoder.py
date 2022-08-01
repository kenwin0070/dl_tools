# -*- coding: UTF-8 -*-

import os
import numpy as np
from tools.tools import PurgedGroupTimeSeriesSplit, set_all_seeds
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, \
    Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import random
import keras_tuner as kt

TRAINING = False
USE_FINETUNE = False
FOLDS = 5
SEED = 42


class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, X, y, splits, batch_size=32, epochs=1, callbacks=None):
        val_losses = []
        for train_indices, test_indices in splits:
            X_train, X_test = [x[train_indices] for x in X], [x[test_indices] for x in X]
            y_train, y_test = [a[train_indices] for a in y], [a[test_indices] for a in y]
            if len(X_train) < 2:
                X_train = X_train[0]
                X_test = X_test[0]
            if len(y_train) < 2:
                y_train = y_train[0]
                y_test = y_test[0]

            model = self.hypermodel.build(trial.hyperparameters)
            hist = model.fit(X_train, y_train,
                             validation_data=(X_test, y_test),
                             epochs=epochs,
                             batch_size=batch_size,
                             callbacks=callbacks)

            val_losses.append([hist.history[k][-1] for k in hist.history])
        val_losses = np.asarray(val_losses)
        self.oracle.update_trial(trial.trial_id,
                                 {k: np.mean(val_losses[:, i]) for i, k in enumerate(hist.history.keys())})
        self.save_model(trial.trial_id, model)


def en():
    """
        打印 文件夹 + 文件名
    """
    for dir_name, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dir_name, filename))

    train_df = pd.read_csv('../input/jane-street-market-prediction/train.csv')
    train_df = train_df.query('date > 85').reset_index(drop=True)
    # limit memory use
    train_df = train_df.astype({c: np.float32 for c in train_df.select_dtypes(include='float64').columns})
    train_df.fillna(train_df.mean(), inplace=True)
    train_df = train_df.query('weight > 0').reset_index(drop=True)
    # train_df['action'] = (train_df['resp'] > 0).astype('int')
    train_df['action'] = ((train_df['resp_1'] > 0) & (train_df['resp_2'] > 0) & (train_df['resp_3'] > 0)
                          & (train_df['resp_4'] > 0) & (train_df['resp'] > 0)).astype('int')
    features = [c for c in train_df.columns if 'feature' in c]

    resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

    X = train_df[features].values
    y = np.stack([(train_df[c] > 0).astype('int') for c in resp_cols]).T  # Multi_target

    f_mean = np.mean(train_df[features[1:]].values, axis=0)

    def create_auto_encoder(input_dim, output_dim, noise=0.05):
        inputs = Input(input_dim)
        encoded = BatchNormalization()(inputs)
        encoded = GaussianNoise(noise)(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        decoded = Dropout(0.2)(encoded)
        decoded = Dense(input_dim, name='decoded')(decoded)
        output = Dense(32, activation='relu')(decoded)
        output = BatchNormalization()(output)
        output = Dropout(0.2)(output)
        output = Dense(output_dim, activation='sigmoid', name='label_output')(output)

        encoder = Model(inputs=inputs, outputs=decoded)
        auto_encoder = Model(inputs=inputs, outputs=[decoded, output])

        auto_encoder.compile(optimizer=Adam(0.001), loss={'decoded': 'mse', 'label_output': 'binary_crossentropy'})
        return auto_encoder, encoder

    def create_model(hp, input_dim, output_dim, encoder):
        inputs = Input(input_dim)
        output = encoder(inputs)
        output = Concatenate()([output, inputs])  # use both raw and encoded features
        output = BatchNormalization()(output)
        output = Dropout(hp.Float('init_dropout', 0.0, 0.5))(output)

        for i in range(hp.Int('num_layers', 1, 3)):
            output = Dense(hp.Int('num_units_{i}', 64, 256))(output)
            output = BatchNormalization()(output)
            output = Lambda(tf.keras.activations.swish)(output)
            output = Dropout(hp.Float(f'dropout_{i}', 0.0, 0.5))(output)
        output = Dense(output_dim, activation='sigmoid')(output)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(hp.Float('lr', 0.00001, 0.1, default=0.001)),
                      loss=BinaryCrossentropy(label_smoothing=hp.Float('label_smoothing', 0.0, 0.1)),
                      metrics=[tf.keras.metrics.AUC(name='auc')])
        return model

    autoencoder, encoder = create_auto_encoder(X.shape[-1], y.shape[-1], noise=0.1)

    set_all_seeds(seed=SEED)
    if TRAINING:
        autoencoder.fit(X, (X, y), epochs=1000, batch_size=4096, validation_split=0.1,
                        callbacks=[EarlyStopping('val_loss', patience=10, restore_best_weights=True)])
        encoder.save_weights('./encoder.hdf5')
    else:
        encoder.load_weights('./encoder.hdf5')

    encoder.trainable = False
    model_fn = lambda hp: create_model(hp, X.shape[-1], y.shape[-1], encoder)

    tuner = CVTuner(hypermodel=model_fn,
                    oracle=kt.oracles.BayesianOptimization(
                        objective=kt.Objective('val_auc', direction='max'),
                        num_initial_points=4,
                        max_trials=20))

    if TRAINING:
        gkf = PurgedGroupTimeSeriesSplit(n_splits=FOLDS, group_gap=20)
        splits = list(gkf.split(y, groups=train_df['date'].values))
        tuner.search((X,), (y,), splits=splits, batch_size=4096, epochs=100,
                     callbacks=[EarlyStopping('val_auc', mode='max', patience=3)])
        hp = tuner.get_best_hyperparameters(1)[0]
        pd.to_pickle(hp, f'./best_hp_{SEED}.pkl')
        for fold, (train_indices, test_indices) in enumerate(splits):
            model = model_fn(hp)
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=4096,
                      callbacks=[EarlyStopping('val_auc', mode='max', patience=10, restore_best_weights=True)])
            model.save_weights(f'./model_{SEED}_{fold}.hdf5')
            model.compile(Adam(hp.get('lr') / 100), loss='binary_crossentropy')
            model.fit(X_test, y_test, epochs=3, batch_size=4096)
            model.save_weights(f'./model_{SEED}_{fold}_finetune.hdf5')
        tuner.results_summary()
    else:
        models = []
        hp = pd.read_pickle(f'../input/v7seedencoderdecoder2223/best_hp_{SEED}.pkl')
        for f in range(FOLDS):
            model = model_fn(hp)
            if USE_FINETUNE:
                model.load_weights(f'../input/v7seedencoderdecoder2223/model_{SEED}_{f}_finetune.hdf5')
            else:
                model.load_weights(f'../input/v7seedencoderdecoder2223/model_{SEED}_{f}.hdf5')
            models.append(model)

        models = models[-2:]
        import janestreet
        env = janestreet.make_env()
        th = 0.5
        for (test_df, pred_df) in tqdm(env.iter_test()):
            if test_df['weight'].item() > 0:
                x_tt = test_df.loc[:, features].values
                if np.isnan(x_tt[:, 1:].sum()):
                    x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
                pred = np.mean([model(x_tt, training=False).numpy() for model in models], axis=0)
                pred = np.median(pred)
                pred_df.action = np.where(pred >= th, 1, 0).astype(int)
            else:
                pred_df.action = 0
            env.predict(pred_df)


if __name__ == "__main__":
    en()