# -*- coding: UTF-8 -*-


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import xgboost as xgb

from tools.tools import reduce_mem_usage

pd.set_option('display.max_columns', 500)
warnings.filterwarnings("ignore")
print("XGBoost version:", xgb.__version__)


def xgboost_model():

    train = pd.read_csv('../input/jane-street-market-prediction/train.csv')
    features = pd.read_csv('../input/jane-street-market-prediction/features.csv')
    example_test = pd.read_csv('../input/jane-street-market-prediction/example_test.csv')
    sample_prediction_df = pd.read_csv('example_sample_submission.csv')
    print("Data is loaded!")

    train, _ = reduce_mem_usage(train)
    exclude = {2, 5, 19, 26, 29, 36, 37, 43, 63, 77, 87, 173, 262, 264, 268, 270, 276, 294, 347, 499}
    train = train[~train.date.isin(exclude)]

    features = [c for c in train.columns if 'feature' in c]

    f_mean = train[features[1:]].mean()
    train[features[1:]] = train[features[1:]].fillna(f_mean)

    train = train[train.weight > 0]

    train['action'] = (train['resp'].values > 0).astype('int')
    train['action1'] = (train['resp_1'].values > 0).astype('int')
    train['action2'] = (train['resp_2'].values > 0).astype('int')
    train['action3'] = (train['resp_3'].values > 0).astype('int')
    train['action4'] = (train['resp_4'].values > 0).astype('int')

    X = train.loc[:, train.columns.str.contains('feature')]
    y = train.loc[:, 'action3'].astype('int').values

    clf2 = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=11,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.7,
        missing=-999,
        random_state=2020,
        tree_method='gpu_hist',  # THE MAGICAL PARAMETER
        reg_alpha=10,
        reg_lambda=10,
    )

    clf2.fit(X, y)

    import janestreet
    env = janestreet.make_env()  # initialize the environment
    iter_test = env.iter_test()  # an iterator which loops over the test set

    tofill = f_mean.values.reshape((1, -1))
    for (test_df, sample_prediction_df) in iter_test:

        if test_df['weight'].values[0] == 0:
            sample_prediction_df.action = 0
        else:
            X_test = test_df.loc[:, features].values
            if np.isnan(X_test.sum()):
                X_test[0, 1:] = np.where(np.isnan(X_test[0, 1:]), tofill, X_test[0, 1:])
            y_preds = int((clf2.predict_proba(X_test)[0][1]) > 0.5)
            sample_prediction_df.action = y_preds
        env.predict(sample_prediction_df)


if __name__ == "__main__":
    xgboost_model()
