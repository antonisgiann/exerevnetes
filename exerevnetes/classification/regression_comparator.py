from exerevnetes.base import BaseComparator
from exereventes.utils.regression_validation import RegressionValidator, default_estimators

from copy import deepcopy

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone


default_metric_funcs = [
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
]

class RegressionComparator(BaseComparator):
    def __init__(
            self,
            X,
            y,
            estimators = None,
            cv = None,
            metric_funcs = None,
            preprocess = None,
            exclude = None
    ):
            if estimators is None:
                estimators = deepcopy(default_estimators)
            if cv is None:
                cv = 5
            if metric_funcs is None:
                metric_funcs = default_metric_funcs

            super().__init__(X, y, estimators, cv, metric_funcs, preprocess, exclude, RegressionValidator)

            if self.exclude:
                for e in self.exclude:
                    self.estimators.pop(e)
            if self.preprocess:
                self.__build_pipelines()

    def __build_pipelines(self):
        """ If ``preprocess`` was given, build pipelines by appending the estimators
        at the end of the preprocessing steps in the ``preprocess``"""
        for est_name, est in self.estimators.items():
            if isinstance(self.preprocess, Pipeline):
                tmp_pipe = clone(self.preprocess)
                if isinstance(tmp_pipe.steps, list):
                    tmp_pipe.steps.append(("model", est))
                if isinstance(tmp_pipe.steps, tuple):
                    tmp_pipe.steps = [tmp_pipe.steps, ("model", est)]
                self.estimators[est_name] = tmp_pipe
            elif isinstance(self.preprocess, ColumnTransformer):
                self.estimators[est_name] = Pipeline(steps=[("preproc", self.preprocess), ("model", est)])
    