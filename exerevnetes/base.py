import time
import inspect
import pandas as pd
from IPython.display import display
from abc import ABC

from .utils import time_format

from sklearn.model_selection import cross_val_predict



class BaseComparator(ABC):
    def __init__(self,
            X,
            y, 
            estimators, 
            cv, 
            metric_funcs, 
            preprocess, 
            exclude,
            Validator
    ):
        self.Validator = Validator
        validate_params = Validator(X, y, estimators, cv, metric_funcs, preprocess, exclude)
        if validate_params.validate():
            self.X = X
            self.y = y
            self.estimators = estimators
            self.cv = cv
            self.metric_funcs = metric_funcs
            self.preprocess = preprocess
            self.exclude = exclude
            self._results = {}

    @classmethod
    def __get_params_names(cls):
        """Get the parameter names of the comparator
        
        Returns
        -------
        list of str
            A sorted list of strings with the names of the parameters given
            to the constructor of the class
        """
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_ins = inspect.signature(init)
        parameters = [
            p for p in init_ins.parameters.values() if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        return sorted([p.name for p in parameters])
    
    @classmethod
    def __format_results(clf, results):
        """Format the results of the comparator from a dictionary to
        a pd.DataFrame with the estimator identifiers as index and 
        the metrics as columns
        
        Returns
        -------
        pd.DataFrame
            The results of the comparator
        """
        return pd.DataFrame(results).T

    def __calculate_scores(self, clf_name, preds):
        for m in self.metric_funcs:
            self._results[clf_name][m.__name__] = m(self.y, preds)
    
    def get_params(self, *args, dataset=False):
        out = dict()
        if args:
            for k in args:
                out[k] = getattr(self, k)
        else:
            for param in self.__get_params_names():
                if not dataset and param in ["X", "y"]:
                    continue
                value = getattr(self, param)
                out[param] = value
        return out

    def get_best(self, metric):
        """ Returns the best-performing estimator based on a specified metric.

        Parameters
        ----------
        metric : str
            The name of the metric used to determine the best estimator.

        Returns
        -------
        best_estimator : estimator object
            The best-performing estimator based on the specified metric
        """
        if not metric in [f.__name__ for f in self.metric_funcs]:
            raise ValueError(f"'{metric}' is not a valid option. Please choose on of f{[f.__name__ for f in self.metric_funcs]}")
        if len(self._results) == 0:
            raise ValueError("There are no models to compare, you need to run the comparator first.")
        if self.preprocess:
            return self.estimators[self._results.sort_values(by=metric).index[-1]].steps[-1][1]
        else:
            return self.estimators[self._results.sort_values(by=metric).index[-1]]
        
    def set_params(self, **kargs):
        valid_params = self.get_params()
        if all(k in valid_params for k in kargs) and self.Validator(**kargs).validate():
            for param, value in kargs.items():
                setattr(self, param, value)
                if param == "exclude":
                    for e in value:
                        self.estimators.pop(e)
        else:
            raise ValueError(
                f"invalid params {param}. "
                f"Valid params {self._get_params_names()!r}"
            )
    
    def get_results(self, sort_by, ascending=False):
        """Return the results of the comparator

        Parameters
        ----------
        sort_by: str
            Name of the metric used to determie the sorting of the results dataframe
        asceding: bool, optional
            Defines the order of the sorting if a sorting is being done

        Returns
        -------
        self._results: pd.DataFrame
            Dataframe containing the results of the comparator
        """
        if len(self._results) == 0:
            raise ValueError("There are no results to be shown, you need to run the comparator first.")
        if sort_by in [f.__name__ for f in self.metric_funcs]:
            return self._results.sort_values(by=sort_by, ascending=ascending)
        else:
            raise ValueError(f"'{sort_by}' is not an available metric. Please choose one of {[f.__name__ for f in self._metric_funcs]}")

    def run(self):
        print(f"The comparator has started...\nRunning for {len(self.estimators)} estimators")
        self._results = {}
        initial_time = time.time()
        for i, (est_name, est) in enumerate(self.estimators.items()):
            print(f"Running cross validation for {i+1}. {est_name}...", end="")
            clf_time = time.time()
            preds = cross_val_predict(est, self.X, self.y, cv=self.cv)
            cv_time = time.time() - clf_time
            self._results[est_name] = {"cv_time": cv_time}
            print((25 - len(est_name))*" ",f"training time: {time_format(cv_time)}", f",   Since beggining: {time_format(time.time() - initial_time)}")
            self.__calculate_scores(est_name, preds)

        # print times and results
        print(f"Total comparator time {time_format(time.time() - initial_time)}")
        self._results = self.__format_results(self._results)
        display(self._results)