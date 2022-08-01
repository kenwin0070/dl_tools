import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from tqdm import tqdm

train = pd.read_csv('../input/jane-street-market-prediction/train.csv')
train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns}) #limit memory use
train.fillna(train.mean(),inplace=True)
train['action'] = (train['resp'] > 0).astype('int')

features = [col for col in train.columns if 'feature' in col]

days = set(train['date'].values)

feature_importance = []
for d in tqdm(days):
    clf = ExtraTreesClassifier()
    X = train[train['date']==d][features].values
    y = train[train['date']==d]['action'].values
    clf.fit(X,y)
    feature_importance.append(clf.feature_importances_)

feature_importance = np.asarray(feature_importance)

fig, ax = plt.subplots(figsize=(10, 100))
ax.matshow(feature_importance)

s_days = sorted(zip(days,[len(train[train['date']==d]) for d in days]),key=lambda x: x[1])
print(s_days[0])

pred = IsolationForest().fit_predict(feature_importance)
outlier_days = [day for day,pred in zip(days,pred) if pred < 0]




if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    print(data)
