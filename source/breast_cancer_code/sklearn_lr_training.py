try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import os

from clearml import Task
from clearml import Logger

params = {
    'solver': 'liblinear',
    'multi_class': 'auto',
    'max_iter': 100,
    'random_state': 1,
    'test_size': 0.2,
    'fit_intercept': True
}

task = Task.init('MASTER-CLASS/Breast_Cancer', 'Scikit-Learn training: LogisticRegression', tags=['scikit-learn'])
task.connect(params)

model_dir = "."
BST_FILE = "model.joblib"

breast_cancer = load_breast_cancer() # https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

y = breast_cancer['target']
X = breast_cancer['data']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params['test_size'],
    random_state=params['random_state'],
    stratify=y
)

model = LogisticRegression(
    solver=params['solver'],
    multi_class=params['multi_class'],
    max_iter=params['max_iter'],
    random_state=params['random_state'],
    fit_intercept=params['fit_intercept']
)  # sklearn LogisticRegression class

model.fit(X_train, y_train)

model_file = os.path.join(model_dir, BST_FILE)

joblib.dump(model, model_file, compress=True)

loaded_model = joblib.load(model_file)

y_pred = loaded_model.predict(X_test)

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
