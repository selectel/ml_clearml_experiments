import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from sklearn.metrics import precision_score, recall_score, f1_score

import os

from clearml import Task
from clearml import Logger

params = {
    'max_depth': 6,
    'eta': 0.1,
    'nthread': 4,
    'num_class': 2,
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'verbosity': 1,
    'random_state': 1,
    'test_size': 0.2
}

task = Task.init('MASTER-CLASS/Breast_Cancer', 'XGBoost training', tags=['xgboost'])
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
