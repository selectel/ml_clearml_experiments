import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score

import pandas as pd

import os

from clearml import Task
from clearml import Logger

params = {
    'objective': 'multiclass',
    'metric': 'softmax',
    'num_class': 3,
    'random_state': 1,
    'test_size': 0.2
}

task = Task.init('MASTER-CLASS/Iris', 'LightGBM training', tags=['lightgbm'])
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


dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val)
dtest = lgb.Dataset(X_test, label=y_test)

lgb_model = lgb.train(
    params={k: v for k, v in params.items() if k not in ['random_state', 'test_size']},
    train_set=dtrain,
    valid_sets=dval,
    valid_names='evaluation'
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
