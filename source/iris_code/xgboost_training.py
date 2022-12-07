import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris

from sklearn.metrics import precision_score, recall_score, f1_score

import pandas as pd

import os

from clearml import Task
from clearml import Logger

params = {
    'max_depth': 6,
    'eta': 0.1,
    'nthread': 4,
    'num_class': 10,
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'verbosity': 3,
    'random_state': 1,
    'test_size': 0.2
}

task = Task.init('MASTER-CLASS/Iris', 'XGBoost training', tags=['xgboost'])
task.connect(params)

model_dir = "."
BST_FILE = "model.bst"

iris = load_iris()

y = iris['target']
X = iris['data']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params['test_size'],
    random_state=params['random_state'],
    stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,
    random_state=params['random_state'],
    stratify=y_train
)  # 0.25 x 0.8 = 0.2

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

xgb_model = xgb.train(
    params={k: v for k, v in params.items() if k not in ['random_state', 'test_size']},
    dtrain=dtrain,
    evals=[(dval, 'evaluation')]
)

model_file = os.path.join(model_dir, BST_FILE)
xgb_model.save_model(model_file)

xgb_model_saved = xgb.Booster(model_file=model_file)

y_pred = xgb_model_saved.predict(dtest)

target_names = [
    'Iris Setosa',
    'Iris Versicolour',
    'Iris Virginica'
]

classification_report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

df = pd.DataFrame(data=classification_report_dict).T

Logger.current_logger().report_table(
    title='Classification report',
    series='',
    iteration=0,
    table_plot=df
)
Logger.current_logger().report_scalar(
    title='precision',
    series='macro',
    value=precision_score(y_test, y_pred, average="macro"),
    iteration=0
)
Logger.current_logger().report_scalar(
    title='recall',
    series='macro',
    value=recall_score(y_test, y_pred, average="macro"),
    iteration=0
)
Logger.current_logger().report_scalar(
    title='f1',
    series='macro',
    value=f1_score(y_test, y_pred, average="macro"),
    iteration=0
)
