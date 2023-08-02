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
    "extra_trees": ExtraTreesClassifier(n_jobs=-1),
    "logistic_regression": LogisticRegression(),
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
    """Compare different classifiers with cross validation 
    on different metrics and aggregate the results in a dataframe

    Parameters
    ----------
    X:  {array-like matrix} of shape (n_samples, n_features)
        Training vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.
        
    y:  array-like of shape (n_samples,)
        Target values.

    classifiers: dictionary of {str: estimator}, default=None (there is predifined
        dictionary of classifiers in that case)
        Contains all the estimators to be compared. ``set_classifiers`` can
        be used to midify that dictionary.

    cv: integer, default=5
        The number of folds for the cross validation

    metric_funcs: list of metric functions that take (y_true, preds), default = None
        (There is a default list of metric functions to be used in that case)
        This list contains the metric functions to be used on the cross validation
        results and the y_true.

    preprocess: sklearn.pipeline.Pipeline or sklearn.compose.ColumnTransformer object, default = None
        It is expected that this object contains only preprocessing steps.
        You can define a different preprocessing by using the ``set_preprocess`` function

    exclude: list of str, default = None
        This list contains names of classifiers to be excluded from the comparator.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassfier, ExtraTreesClassifier
    >>> from sklearn.metrics import ruc_auc_score
    >>> from sklearn.pipeline import Pipeline
    >>> from exerevnetes import BinaryClassifierComparator
    >>> X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=47)

    Run with the default parameters

    >>> cmp = BinaryClassifierComparator(X, y)
    >>> cmp.run()
    >>> cmp.get_results().shape
    (7,6) 
    >>> cmp.get_best_clf()
    
    Define the classifiers of your interest

    >>> cmp.set_classifiers({"forest": RandomForestClassifier(), "extra": ExtraTreesClassifier()})
    >>> cmp.run()

    Define your own metrics or/and exclude a default classifier from your comparison

    >>> def my_function(y_true, preds):
    ...     something
    ...     something
    ...     return score
    >>> cmp = BinaryClassifierComparator(X, y, metric_funcs = [my_function, roc_auc_score], exclude = ["svc"])
    >>> cmp.run()

    Add a pipeline, or change the pipeline being used
    >>> my_pipe = Pipeline(steps=[("step1", step1), ("step2", step2)])
    >>> cmp.set_preprocess(my_pipe)
    >>> cmp.run()
    """
    def __init__(
            self, 
            X, 
            y, 
            classifiers: dict = None, 
            cv: int = 5, 
            metric_funcs: list = default_metric_funcs, 
            preprocess=None, 
            exclude: list = None
    ):
        if not isinstance(X, np.ndarray) and not hasattr(X, "values"):
            raise TypeError("'X' is not a numpy.array nor a has a 'values' attribute to return a numpy.array.")
        if not isinstance(y, np.ndarray) and not hasattr(y, "values"):
            raise TypeError("'y' is not a numpy.array nor a has a 'values' attribute to return a numpy.array.")
        if classifiers and not isinstance(classifiers, dict):
            raise TypeError("'classifiers' must be a dictionary.")
        if isinstance(cv, int) and cv < 2:
            raise ValueError("cv must be a positive integer greater than 1.")
        if not isinstance(metric_funcs, list) and not all(callable(m_f) for m_f in metric_funcs):
            raise TypeError("'metric_funcs' must be a list of callable functions.")
        if classifiers:
            if exclude:
                raise AttributeError("You can't use the 'exclude' attribute when passing 'classifiers'")
        else:
            if exclude and (not isinstance(exclude, list) or not all(ex in default_classifiers for ex in exclude)):
                raise ValueError(f"'exclude' must be a list of available default classifiers. All default classifiers are: {list(default_classifiers.keys())}")
            elif exclude and (len(set(exclude)) != len(exclude)):
                raise ValueError("'exlucde' must contain unique classifiers.")

        if len(X) == 0:
            raise ValueError("An empty array was given for 'X'")
        if len(y) != len(X):
            raise ValueError(f"There is a mismatch between 'X' and 'y'. 'X' has shape {X.shape} and 'y' has shape {y.shape}")
        if classifiers and len(classifiers) == 0:
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
        if classifiers:
            self.classifiers = classifiers
        else:
            self.classifiers = default_classifiers
        self.cv = cv
        self.metric_funcs = metric_funcs
        self.preprocess = preprocess
        self._results = {}
        self.exclude = exclude
        
        if self.exclude:
            for e in self.exclude:
                self.classifiers.pop(e)
        if self.preprocess:
            self.__build_pipelines(self.preprocess)
    
    def run(self):
        print(f"The comparator has started...\nRunning for {len(self.classifiers)} classifiers")
        self._results = {}
        initial_time = time.time()
        for i, (clf_name, clf) in enumerate(self.classifiers.items()):
            print(f"Running cross validation for {i+1}. {clf_name}...", end="")
            clf_time = time.time()
            preds = cross_val_predict(clf, self.X, self.y, cv=self.cv)
            cv_time = time.time() - clf_time
            self._results[clf_name] = {"cv_time": cv_time}
            print((25 - len(clf_name))*" ",f"training time: {time_format(cv_time)}", f",   Since beggining: {time_format(time.time() - initial_time)}")
            self.__calculate_scores(clf_name, preds)

        # print times and results
        print(f"Total comparator time {time_format(time.time() - initial_time)}")
        self.__format_results()
        display(self._results)

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
            self._results[clf_name][m.__name__] = m(self.y, preds)

    def __format_results(self):
        self._results = pd.DataFrame(self._results).T

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
        
    def get_results(self, sort_by=None, ascending=False):
        if len(self._results) == 0:
            raise ValueError("There are no results to be shown, you need to run the comparator first.")
        if sort_by:
            if sort_by in [f.__name__ for f in self.metric_funcs]:
                return self._results.sort_values(by=sort_by, ascending=ascending)
            else:
                raise ValueError(f"'{sort_by}' is not an available metric. Please choose one of {[f.__name__ for f in self.metric_funcs]}")
        else:
            return self._results
            
    def get_preprocess(self):
        if not self.preprocess:
            warnings.warn("The preprocess attribute is \033[1mNone\033[0m.", UserWarning)
        return self.preprocess
    
    def get_classifiers(self):
        return self.classifiers

    def get_best_clf(self, metric="f1_score"):
        if len(self._results) == 0:
            raise ValueError("There are no models to compare, you need to run the comparator first.")
        if self.preprocess:
            return self.classifiers[self._results.sort_values(by=metric).index[-1]].steps[-1][1]
        else:
            return self.classifiers[self._results.sort_values(by=metric).index[-1]]