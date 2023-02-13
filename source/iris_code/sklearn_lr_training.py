try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pandas as pd

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

task = Task.init('DEMO/Iris 1', 'Scikit-Learn training: LogisticRegression', tags=['scikit-learn'])
task.connect(params)

model_dir = "."
BST_FILE = "model.joblib"

iris = load_iris()

y = iris['target']
X = iris['data']

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
