# -*- coding: UTF-8 -*-


def bayesian_optimize():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Bidirectional, GRU
    from hyperopt import hp, fmin, tpe, Trials
    from hyperopt.early_stop import no_progress_loss

    def dnn_optimize():
        # 模型
        def model(params):
            model = Sequential()
            model.add(Bidirectional(GRU(units=int(params['units']), activation='relu'),
                                    input_shape=(x_train.shape[1], x_train.shape[-1])))
            model.add(Dense(1))
            model.summary()
            model.compile(loss='mae', optimizer='adam')
            model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val, y_val))
            # 用测试集上的损失进行寻优
            score_loss = model.evaluate(x_val, y_val, verbose=2)
            print("Text loss:", score_loss)
            return score_loss

        def create_dataset(dataset, look_back=8):
            data_x, data_y = [], []
            for i in range(len(dataset) - look_back - 1):
                # 获取全部元素
                x = dataset[i:i + look_back, 0:dataset.shape[1]]
                data_x.append(x)
                # 第0列，想要预测的数据
                y = dataset[i + look_back, 0]
                data_y.append(y)
            return np.array(data_x), np.array(data_y)

        # 定义参数优化函数
        def param_hyperopt(max_evals=100):
            trials = Trials()
            # 提前停止条件
            early_stop_fn = no_progress_loss(20)
            # 优化模型
            params_best = fmin(fn=model, space=params_space, algo=tpe.suggest, max_evals=max_evals,
                               trials=trials, early_stop_fn=early_stop_fn)
            print('best params:', params_best)
            return params_best, trials

        # 定义参数模型空间
        params_space = {
            "units": hp.quniform('units', 40, 120, 4)
        }

        data1 = pd.read_csv('201701.csv', usecols=['站点名称', 'PM2.5', 'PM10'], engine='python')
        data2 = pd.read_csv('201702.csv', usecols=['站点名称', 'PM2.5', 'PM10'], engine='python')
        data3 = pd.read_csv('201703.csv', usecols=['站点名称', 'PM2.5', 'PM10'], engine='python')
        data4 = pd.read_csv('201704.csv', usecols=['站点名称', 'PM2.5', 'PM10'], engine='python')
        data5 = pd.read_csv('201705.csv', usecols=['站点名称', 'PM2.5', 'PM10'], engine='python')
        data6 = pd.read_csv('201706.csv', usecols=['站点名称', 'PM2.5', 'PM10'], engine='python')
        data = pd.concat([data1, data2, data3, data4, data5, data6])
        data = data[data['站点名称'] == '衡水监测站']
        data = pd.concat([data], ignore_index=True)
        del data['站点名称']
        print(data[:10])
        print(data.shape)  # (2880, 2)

        # 输入，输出划分
        data = data[['PM2.5', 'PM10']]
        print(data[0:10])

        # 归一化
        scale = MinMaxScaler()
        data = scale.fit_transform(data)
        print(data[0:10])

        # 划分数据集
        X_data = data[0:int(len(data) * 0.9)]
        Y_data = data[int(len(data) * 0.9):]
        # 训练集
        x_train, y_train = create_dataset(X_data)
        x_val, y_val = create_dataset(Y_data)
        print(x_train.shape)  # (3900, 8, 2)
        print(x_val.shape)
        # 调用参数寻优函数
        params_best, trials = param_hyperopt(20)

    def xgboost_optimize():
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import KFold
        from sklearn.metrics import roc_auc_score
        import pandas as pd
        import time
        import datetime
        import hyperopt

        # set training
        train_data = pd.read_csv("train.csv")
        X_train = train_data[train_data.columns[2:]]
        Y_train = train_data[train_data.columns[1]]

        # no need for use of test data
        train_data = pd.read_csv("train.csv")
        X_test = train_data[train_data.columns[1:]]

        def train_model(X, X_test, y, params=None, n_fold=5, plot_feature_importance=False, model=None):
            oof = np.zeros(len(X))
            prediction = np.zeros(len(X_test))
            folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
            scores = []
            feature_importance = pd.DataFrame()
            for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
                print('Fold', fold_n, 'started at', time.ctime())
                X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
                y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

                train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
                valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

                watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
                model = xgb.train(dtrain=train_data, num_boost_round=40000, evals=watchlist, early_stopping_rounds=200,
                                  verbose_eval=500, params=params)
                y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                             ntree_limit=model.best_ntree_limit)
                # use this when you want predict test samples y_pred = model.predict(xgb.DMatrix(X_test,
                # feature_names=X.columns), ntree_limit=model.best_ntree_limit)
                y_pred = prediction

                oof[valid_index] = y_pred_valid.reshape(-1, )
                scores.append(roc_auc_score(np.where(y_pred_valid > 0.5, 1, 0), y_valid))

                prediction += y_pred

            prediction /= n_fold

            print('CV roc_curve: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

            return oof, prediction, model, np.mean(scores)

        def xgb_model(params):
            """used for hyperopt"""

            n_fold = params['n_fold'] + 5
            xgb_params = {'eta': 1 / params['eta'],
                          'max_depth': params['max_depth'] + 5,
                          'subsample': params['subsample'],
                          'objective': 'reg:linear',
                          'eval_metric': 'error',
                          'silent': True,
                          'nthread': 7}
            _, _, _, score = train_model(X_train, X_test, Y_train, xgb_params, n_fold=n_fold)
            return -score

        # start training
        xgb_params = {'n_fold': 0,
                      'eta': 100.0 / 3.0,
                      'max_depth': 5,
                      'subsample': 0.9}
        xgb_model(xgb_params)

        # start training with automatic parameter turning
        space = {'n_fold': hyperopt.hp.randint('n_fold', 10),
                 'eta': hyperopt.hp.uniform('eta', 10, 10000),
                 'max_depth': hyperopt.hp.randint('max_depth', 45),
                 'subsample': hyperopt.hp.uniform('subsample', 0.6, 0.999)}
        # algo = hyperopt.partial(hyperopt.tpe.suggest, n_startup_jobs=10)
        # best = hyperopt.fmin(xgb_model, space, algo , 200)
        best = hyperopt.fmin(xgb_model, space, hyperopt.tpe.suggest, 200)
        print(best)

        best = {'eta': 2640.346312547211, 'max_depth': 32, 'n_fold': 8, 'subsample': 0.638448231111887}
        xgb_model(best)


if __name__ == '__main__':
    print('PyCharm')
