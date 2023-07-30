import time
import warnings
import pandas as pd
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
        self.__check_constructor_args(locals())
        self.X = X
        self.n_classes = len(Counter(y))
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
            print((15 - len(clf_name))*" ",f"training time: {time_format(cv_time)}")
            self.__calculate_scores(clf_name, preds)

        # print times and metrics
        print(f"Total comparator time {time_format(time.time() - initial_time)}")
        self.__format_metrics()
        display(self._metrics)

    def __check_constructor_args(self, args):
        if len(args["X"]) == 0:
            raise ValueError("An empty array was given for 'X'")
        if len(args["y"]) != len(args["X"]):
            raise ValueError(f"There is a mismatch between 'X' and 'y'. 'X' has shape {args['X'].shape} and 'y' has shape {args['y'].shape}")
        if len(args["classifiers"]) == 0:
            arg_type = str(type(args['classifiers'])).split()[-1][1:-2]
            if arg_type != "dict":
                type_msg = "Be aware that 'classifiers' must be a dictionary"
            else:
                type_msg = ""
            raise ValueError(f"An empty {arg_type} was given for 'classifiers'. {type_msg}")
        if len(args["metric_funcs"]) == 0:
            arg_type = str(type(args['metric_funcs'])).split()[-1][1:-2]
            if arg_type != "list":
                type_msg = "Be aware that 'metric_funcs' must be a list"
            else:
                type_msg = ""
            raise ValueError(f"An empty {arg_type} was given for 'metric_funcs'. {type_msg}")
        n_classes = Counter(args['y'])
        if len(n_classes) != 2:
            raise ValueError(f"Expected 2 classes but got {n_classes}.")
        self.__preprocess_check(args['preprocess'])

    def __preprocess_check(self, preprocess):
        if preprocess == None:
            return True
        if type(preprocess) != Pipeline:
                raise AttributeError(f"The comparator accepts only \033[34msklearn.pipeline.Pipeline\033[0m objects as 'preprocess'.")
        elif hasattr(preprocess.steps[-1][1], "predict"):
                raise AttributeError("The 'preprocess' contains a predictor. Please make sure 'preprocess' contains only preprocessing steps")
        else:
            return True
        
    def __calculate_scores(self, clf_name, preds):
        for m in self.metric_funcs:
            self._metrics[clf_name][m.__name__] = m(self.y, preds)

    def __format_metrics(self):
        self._metrics = pd.DataFrame(self._metrics).T

    def __build_pipelines(self, preprocess):
        for clf_name, clf in self.classifiers.items():
            tmp_pipe = clone(preprocess)
            tmp_pipe.steps.append(("model", clf))
            self.classifiers[clf_name] = tmp_pipe

    def set_preprocess(self, preprocess):
        if preprocess == None:
            raise ValueError("'preprocess' is None")
        if self.__preprocess_check(preprocess):
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

    def get_best_clf(self, metric="f1_score"):
        if len(self._metrics) == 0:
            raise ValueError("There are no models to compare, you need to run the comparator first.")
        if self.preprocess:
            return self.classifiers[self._metrics.sort_values(by=metric).index[-1]].steps[-1][1]
        else:
            return self.classifiers[self._metrics.sort_values(by=metric).index[-1]]