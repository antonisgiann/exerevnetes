import pytest
import numpy as np
import pandas as pd

from .._clf_comparator import BinaryClassifierComparator
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


np.random.seed(47)

# General testing dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_classes=2, random_state=47)

####### General test #######
@pytest.mark.parametrize("X, y, classifiers, cv, metric_funcs, num_pipe, expected_shape", [
    (
        X, # dependent variables
        y, # independent variables
        {"forest": RandomForestClassifier(), "extraTrees": ExtraTreesClassifier()}, # classifiers
        7, # cross validation folds
        [f1_score, roc_auc_score, accuracy_score], # results
        Pipeline(steps=[("scaler", StandardScaler()), ("pca", PCA())]), # preprocessing pipeline
        (2,4) # shape of the results output
    )
])
def test_class(X, y, classifiers, cv, metric_funcs, num_pipe, expected_shape):
    """testing all class attributes together"""
    X = pd.DataFrame(X)
    preprocess = ColumnTransformer([("num_proc", num_pipe, X.columns.tolist())])
    cmp = BinaryClassifierComparator(X, y, classifiers, cv, metric_funcs, preprocess)
    cmp.run()
    assert cmp.get_results(sort_by=metric_funcs[0].__name__).shape == expected_shape
    assert isinstance(cmp.get_best_clf(), BaseEstimator)
    tmp = cmp.get_preprocess()
    assert isinstance(tmp, (Pipeline, ColumnTransformer))
    if isinstance(tmp, Pipeline):
        assert(len(tmp.steps)) == 2
    if isinstance(tmp, ColumnTransformer):
        assert(len(tmp.transformers[0][1].steps) == 2)


####### tests #######
@pytest.mark.parametrize("X, y", [
    (np.array([[0,0,0,1],[0,0,1,0]]), np.array([[1,0]]).ravel())
    ])
def test_small_sample_size(X, y):
    """test that comparator fails for small sample size because of cross-validation"""
    cmp = BinaryClassifierComparator(X, y)
    with pytest.raises(ValueError) as exce:
        cmp.run()


@pytest.mark.parametrize("X, y, cv, expected", [
    (np.random.randint(0, 100, size=1000).reshape(100,10), np.random.randint(0, 2, size=100).ravel(), 5, (7,6))
    ])
def test_default_results_size(X, y, cv, expected):
    """test the size of the results dataframe"""
    cmp = BinaryClassifierComparator(X, y, cv=cv)
    cmp.run()
    assert cmp.get_results().shape == expected


@pytest.mark.parametrize("X, y",[
        (np.random.randint(0, 100, size=1000).reshape(100,10), np.random.randint(0, 1, size=100)),
        (np.random.randint(0, 100, size=1000).reshape(100,10), np.random.randint(0, 5, size=100))
    ])
def test_n_class(X, y):
    """test if there are two classes in y_true."""
    with pytest.raises(ValueError) as exce:
        cmp = BinaryClassifierComparator(X, y)
    assert str(exce.value).startswith("Expected 2 classes but got")


def test_if_run():
    """test if comparator was run before getting the best classifier"""
    cmp = BinaryClassifierComparator(X, y)
    with pytest.raises(ValueError) as exce:
        cmp.get_best_clf()
    assert str(exce.value) == "There are no models to compare, you need to run the comparator first."
    with pytest.raises(ValueError) as exce:
        cmp.get_results()
    assert str(exce.value) == "There are no results to be shown, you need to run the comparator first."


@pytest.mark.parametrize("metric_funcs, expected", [([f1_score, roc_auc_score], 3), ([roc_auc_score], 2)])
def test_number_of_results(metric_funcs, expected):
    """testing the number of results"""
    cmp = BinaryClassifierComparator(X, y, metric_funcs=metric_funcs)
    cmp.run()
    assert cmp.get_results().shape[1] == expected


def test_multiple_run_calls():
    """testing multiple calls of run()"""
    cmp = BinaryClassifierComparator(X, y)
    cmp.run()
    cmp.run()
    cmp.run()
    assert cmp.get_results().dropna().shape == (7,6)


@pytest.mark.parametrize("classifiers, expected", [
    ({"forest": RandomForestClassifier()}, (1,6)),
    ({"forest": RandomForestClassifier(), "extra_tree": ExtraTreesClassifier()}, (2,6))
])
def test_classifiers_setter(classifiers, expected):
    """test the classifiers setter function"""
    cmp = BinaryClassifierComparator(X, y, classifiers)
    cmp.run()
    assert cmp.get_results().shape == expected
    assert cmp.get_classifiers() == classifiers


@pytest.mark.parametrize("exclude", [(["random_forest", "random_forest"])])
def test_exclude_uniqueness(exclude):
    """test if exclude contains unique values"""
    with pytest.raises(ValueError) as exce:
        cmp = BinaryClassifierComparator(X, y, exclude=exclude)


####### Preprocess tests #######
@pytest.mark.parametrize("classifiers, expected", [
        ({"pipe":Pipeline(steps=[("scaler", StandardScaler()), ("model", RandomForestClassifier(n_jobs=-1))])}, 1)
    ])
def test_pipelines_as_classifiers(classifiers, expected):
    """test the output of passing a single pipeline for classifiers"""
    cmp = BinaryClassifierComparator(X, y, classifiers=classifiers)
    cmp.run()
    assert cmp.get_results().shape[0] == expected


@pytest.mark.parametrize("preprocess, expected", [
    (Pipeline(steps=[("scaler", StandardScaler())]), (7,6))
])
def test_preprocess_attr_results(preprocess, expected):
    """testing the preprocess attribute"""
    cmp = BinaryClassifierComparator(X, y, preprocess=preprocess)
    cmp.run()
    assert cmp.get_results().shape == expected


@pytest.mark.parametrize("preprocess", [
    (None),
    (Pipeline(steps=[("scaler", StandardScaler())]))
])
def test_preprocess_attr_best_clf(preprocess):
    """testing best_clf function with preprocess attribute"""
    cmp = BinaryClassifierComparator(X, y, preprocess=preprocess)
    cmp.run()
    assert hasattr(cmp.get_best_clf(), "predict")


@pytest.mark.parametrize("preprocess", [(Pipeline(steps=[("scaler", StandardScaler())]))])
def test_get_preprocess(preprocess):
    """testing the get_preprocess() function"""
    cmp = BinaryClassifierComparator(X, y, preprocess=preprocess)
    assert cmp.get_preprocess() == preprocess
    assert isinstance(cmp.get_preprocess(), (Pipeline, ColumnTransformer))


def test_if_preprocess_pipeline():
    """testing if preprocess is an sklearn.pipeline.Pipeline"""
    preprocess = lambda x: x**2
    with pytest.raises(AttributeError) as exce:
        cmp = BinaryClassifierComparator(X, y, preprocess=preprocess)


@pytest.mark.parametrize("preprocess", [
    (Pipeline(steps=[("scaler", StandardScaler()), ("model", RandomForestClassifier())]))
])
def test_if_preprocess_contains_predictor(preprocess):
    """testing if preprocessing contains a predictor"""
    with pytest.raises(AttributeError) as exce:
        cmp = BinaryClassifierComparator(X, y, preprocess=preprocess)


@pytest.mark.parametrize("preprocess", [
    (Pipeline(steps=[("scaler", StandardScaler()), ("PCA", PCA())])),
])
def test_preprocess_setter(preprocess):
    """testing preprocess setter and getter"""
    cmp = BinaryClassifierComparator(X, y)
    cmp.run()
    cmp.set_preprocess(preprocess)
    assert cmp.get_preprocess() == preprocess
    cmp.run()
    cmp.get_best_clf()

####### Other tests #######
@pytest.mark.parametrize("exclude", [(["svc", "extra_trees"])])
def test_exclude_attr(exclude):
    cmp = BinaryClassifierComparator(X, y, exclude=exclude)
    cmp.run()
    assert len(cmp.get_classifiers()) == 5
    assert cmp.get_results().shape == (5,6)


@pytest.mark.parametrize("exclude", [(["svc", "extra_trees"])])
def test_exlcude_setter(exclude):
    """test that you can exclude classifiers after initialization"""
    cmp = BinaryClassifierComparator(X, y, exclude=None)
    assert len(cmp.get_classifiers()) == 7
    cmp.set_exclude(exclude=exclude)
    cmp.run()
    assert len(cmp.get_classifiers()) == 5