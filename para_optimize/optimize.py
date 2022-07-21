# coding=utf-8

def bayesian_optimize():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Bidirectional, GRU
    from hyperopt import hp, fmin, tpe, Trials
    from hyperopt.early_stop import no_progress_loss

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


if __name__ == '__main__':
    print('PyCharm')