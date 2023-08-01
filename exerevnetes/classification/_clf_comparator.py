import time
import warnings
import pandas as pd
import numpy as np
from collections import Counter
from IPython.display import display

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from exerevnetes.utils import time_format


default_classifiers = {
    "random_forest": RandomForestClassifier(n_jobs=-1),
    "extra_tree": ExtraTreesClassifier(n_jobs=-1),
    "logistic_reg": LogisticRegression(),
    "svc": SVC(),
    "catboost": CatBoostClassifier(verbose=False, allow_writing_files=False),
    "xgboost": XGBClassifier(),
    "naive_bayes": GaussianNB()
}

default_metric_funcs = [
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    accuracy_score
]

class BinaryClassifierComparator:
    def __init__(self, X, y, classifiers: dict = default_classifiers, cv=5, metric_funcs: list = default_metric_funcs, preprocess=None):
        if not isinstance(X, np.ndarray) and not hasattr(X, "values"):
            raise TypeError("'X' is not a numpy.array nor a has a 'values' attribute to return a numpy.array.")
        if not isinstance(y, np.ndarray) and not hasattr(y, "values"):
            raise TypeError("'y' is not a numpy.array nor a has a 'values' attribute to return a numpy.array.")
        if not isinstance(classifiers, dict):
            raise TypeError("'classifiers' must be a dictionary.")
        if isinstance(cv, int) and cv < 2:
            raise ValueError("cv must be a positive integer greater than 1.")
        if not isinstance(metric_funcs, list) and not all(callable(m_f) for m_f in metric_funcs):
            raise TypeError("'metric_funcs' must be a list of callable functions.")

        if len(X) == 0:
            raise ValueError("An empty array was given for 'X'")
        if len(y) != len(X):
            raise ValueError(f"There is a mismatch between 'X' and 'y'. 'X' has shape {X.shape} and 'y' has shape {y.shape}")
        if len(classifiers) == 0:
            raise ValueError(f"An empty {str(type(classifiers)).split()[-1][1:-2]} was given for 'classifiers'.")
        if len(metric_funcs) == 0:
            raise ValueError(f"An empty {str(type(metric_funcs)).split()[-1][1:-2]} was given for 'metric_funcs'.")
        n_classes = Counter(y)
        if len(n_classes) != 2:
            raise ValueError(f"Expected 2 classes but got {n_classes}.")
        if preprocess:
            self.__preprocess_checks(preprocess)

        self.X = X
        self.n_classes = n_classes
        self.y = y
        self.classifiers = classifiers
        self.cv = cv
        self.metric_funcs = metric_funcs
        self.preprocess = preprocess
        self._metrics = {}
        if self.preprocess:
            self.__build_pipelines(self.preprocess)
    
    def run(self):
        print(f"The comparator has started...\nRunning for {len(self.classifiers)} classifiers")
        self._metrics = {}
        initial_time = time.time()
        for i, (clf_name, clf) in enumerate(self.classifiers.items()):
            print(f"Running cross validation for {i+1}. {clf_name}...", end="")
            clf_time = time.time()
            preds = cross_val_predict(clf, self.X, self.y, cv=self.cv)
            cv_time = time.time() - clf_time
            self._metrics[clf_name] = {"cv_time": cv_time}
            print((15 - len(clf_name))*" ",f"training time: {time_format(cv_time)}", f",   Since beggining: {time_format(time.time() - initial_time)}")
            self.__calculate_scores(clf_name, preds)

        # print times and metrics
        print(f"Total comparator time {time_format(time.time() - initial_time)}")
        self.__format_metrics()
        display(self._metrics)

    def __preprocess_checks(self, preprocess):
        if not isinstance(preprocess, (Pipeline, ColumnTransformer)):
                raise AttributeError(f"The comparator accepts only \033[34msklearn.pipeline.Pipeline\033[0m or \033[34msklearn.compose.ColumnTransformer\033[0m objects as 'preprocess'.")
        if isinstance(preprocess, Pipeline):
            if hasattr(preprocess.steps[-1][1], "predict"):
                raise AttributeError("The 'preprocess' contains a predictor. Please make sure 'preprocess' contains only preprocessing steps")
        elif isinstance(preprocess, ColumnTransformer):
            if hasattr(preprocess.transformers[-1][1], "predict"):
                raise AttributeError("The 'preprocess' contains a predictor. Please make sure 'preprocess' contains only preprocessing steps")
        
        return True
        
    def __calculate_scores(self, clf_name, preds):
        for m in self.metric_funcs:
            self._metrics[clf_name][m.__name__] = m(self.y, preds)

    def __format_metrics(self):
        self._metrics = pd.DataFrame(self._metrics).T

    def __build_pipelines(self, preprocess):
        for clf_name, clf in self.classifiers.items():
            if isinstance(preprocess, Pipeline):
                tmp_pipe = clone(preprocess)
                tmp_pipe.steps.append(("model", clf))
                self.classifiers[clf_name] = tmp_pipe
            elif isinstance(preprocess, ColumnTransformer):
                self.classifiers[clf_name] = Pipeline(steps=[("preproc", preprocess), ("model", clf)])

    def set_classifiers(self, classifiers):
        if classifiers == None:
            raise ValueError("'classifiers' is None.")
        if not isinstance(classifiers, dict):
            raise TypeError("'classifiers' must be a dictionary.")
        self.classifiers = classifiers
        if self.preprocess:
            self.__build_pipelines(self.preprocess)

    def set_preprocess(self, preprocess):
        if preprocess == None:
            raise ValueError("'preprocess' is None.")
        if self.__preprocess_checks(preprocess):
            self.preprocess = preprocess
            self.__build_pipelines(self.preprocess)
        
    def get_metrics(self):
        if len(self._metrics) == 0:
            raise ValueError("There are no metrics to be shown, you need to run the comparator first.")
        return self._metrics
    
    def get_preprocess(self):
        if not self.preprocess:
            warnings.warn("The preprocess attribute is \033[1mNone\033[0m.", UserWarning)
        return self.preprocess
    
    def get_classifiers(self):
        return self.classifiers

    def get_best_clf(self, metric="f1_score"):
        if len(self._metrics) == 0:
            raise ValueError("There are no models to compare, you need to run the comparator first.")
        if self.preprocess:
            return self.classifiers[self._metrics.sort_values(by=metric).index[-1]].steps[-1][1]
        else:
            return self.classifiers[self._metrics.sort_values(by=metric).index[-1]]