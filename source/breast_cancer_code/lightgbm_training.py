import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score

import os

from clearml import Task
from clearml import Logger

params = {
    'max_depth': 6,
    'eta': 0.1,
    'nthread': 4,
    'num_class': 2,
    'objective': 'multiclass',
    'metric': 'softmax',
    'verbosity': 3,
    'random_state': 1,
    'test_size': 0.2
}

task = Task.init('MASTER-CLASS/Breast_Cancer', 'LightGBM training', tags=['lightgbm'])
task.connect(params)

model_dir = "."
BST_FILE = "model.bst"

breast_cancer = load_breast_cancer()

y = breast_cancer['target']
X = breast_cancer['data']

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

Logger.current_logger().report_scalar(
    title='precision',
    series='binary',
    value=precision_score(y_test, y_pred, average="binary"),
    iteration=0
)
Logger.current_logger().report_scalar(
    title='recall',
    series='binary',
    value=recall_score(y_test, y_pred, average="binary"),
    iteration=0
)
Logger.current_logger().report_scalar(
    title='f1',
    series='binary',
    value=f1_score(y_test, y_pred, average="binary"),
    iteration=0
)
