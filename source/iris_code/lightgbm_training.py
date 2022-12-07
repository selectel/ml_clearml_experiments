import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from clearml import Task

import os

Task.init('MASTER-CLASS/Iris', 'LightGBM training')

model_dir = "."
BST_FILE = "model.bst"

iris = load_iris()

y = iris['target']
X = iris['data']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2


dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val)
dtest = lgb.Dataset(X_test, label=y_test)

params = {
    'objective': 'multiclass',
    'metric': 'softmax',
    'num_class': 3
}

lgb_model = lgb.train(
    params=params,
    train_set=dtrain,
    valid_sets=dval
)

model_file = os.path.join(model_dir, BST_FILE)

lgb_model.save_model(model_file)

lgb_model_saved = lgb.Booster(model_file=model_file)

y_pred = lgb_model_saved.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)

target_names = [
    'Iris Setosa',
    'Iris Versicolour',
    'Iris Virginica'
]

print(classification_report(y_test, y_pred, target_names=target_names))
