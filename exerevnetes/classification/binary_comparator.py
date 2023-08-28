from exerevnetes.base import BaseComparator
from exerevnetes.utils.binary_classification_validation import BinaryClassificationValidator, default_estimators

from copy import deepcopy

from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone


default_metric_funcs = [
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    accuracy_score
]

class BinaryClassificationComparator(BaseComparator):
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
        Contains all the estimators to be compared. ``set_params` can
        be used to midify that dictionary.

    cv: integer, default=5
        The number of folds for the cross validation

    metric_funcs: list of metric functions that take (y_true, preds), default = None
        (There is a default list of metric functions to be used in that case)
        This list contains the metric functions to be used on the cross validation
        results and the y_true.

    preprocess: sklearn.pipeline.Pipeline or sklearn.compose.ColumnTransformer object, default = None
        It is expected that this object contains only preprocessing steps.
        You can define a different preprocessing by using the ``set_params`` function

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
    >>> cmp._results.shape
    (7,6) 
    >>> cmp.get_best(metric="f1_score") # return the estimator with the highest metric
    
    Define the classifiers of your interest

    >>> cmp.set_params(estimators={"forest": RandomForestClassifier(), "extra": ExtraTreesClassifier()})
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
    >>> cmp.set_params(preprocess=my_pipe)
    >>> cmp.run()
    """
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

            super().__init__(X, y, estimators, cv, metric_funcs, preprocess, exclude, BinaryClassificationValidator)

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
                elif isinstance(tmp_pipe.steps, tuple):
                    tmp_pipe.steps = [tmp_pipe.steps, ("model", est)]
                self.estimators[est_name] = tmp_pipe
            elif isinstance(self.preprocess, ColumnTransformer):
                self.estimators[est_name] = Pipeline(steps=[("preproc", self.preprocess), ("model", est)])
    