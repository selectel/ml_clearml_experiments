import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris

from clearml import Task

import os

Task.init('MASTER-CLASS/Iris', 'XGBoost training')

model_dir = "."
BST_FILE = "model.bst"

iris = load_iris()

y = iris['target']
X = iris['data']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 6,
    'eta': 0.1,
    'silent': 1,
    'nthread': 4,
    'num_class': 10,
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'verbosity': -1
}

xgb_model = xgb.train(params=param, dtrain=dtrain)

model_file = os.path.join((model_dir), BST_FILE)
xgb_model.save_model(model_file)

xgb_model_saved = xgb.Booster(model_file=model_file)

y_pred = xgb_model_saved.predict(dtest)

target_names = [
    'Iris Setosa',
    'Iris Versicolour',
    'Iris Virginica'
]

print(classification_report(y_test, y_pred, target_names=target_names))
