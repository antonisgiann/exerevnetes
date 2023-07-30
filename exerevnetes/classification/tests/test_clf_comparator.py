import pytest
import numpy as np

from .._clf_comparator import BinaryClassifierComparator
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


np.random.seed(47)

# General testing dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=47)


####### General tests #######
@pytest.mark.parametrize(
        "X, y",
        [(np.array([[0,0,0,1],[0,0,1,0]]), np.array([[1,1,1,0]]).ravel())]
)
def test_small_sample_size(X, y):
    """test that comparator fails for small sample size because of cross-validation"""
    cmp = BinaryClassifierComparator(X, y)
    with pytest.raises(ValueError) as exce:
        cmp.run()


@pytest.mark.parametrize(
    "X, y, cv, expected", 
    [(np.random.randint(0, 100, size=1000).reshape(100,10), np.random.randint(0, 2, size=100).ravel(), 5, (7,6))]
)
def test_default_metrics_size(X, y, cv, expected):
    """test the size of the metrics dataframe"""
    cmp = BinaryClassifierComparator(X, y, cv=cv)
    cmp.run()
    assert cmp.get_metrics().shape == expected


@pytest.mark.parametrize(
    "X, y",
    [
        (np.random.randint(0, 100, size=1000).reshape(100,10), np.random.randint(0, 1, size=100)),
        (np.random.randint(0, 100, size=1000).reshape(100,10), np.random.randint(0, 5, size=100))
    ]
)
def test_n_class(X, y):
    """test if there are two classes in y_true."""
    with pytest.raises(ValueError) as exce:
        cmp = BinaryClassifierComparator(X, y)
    assert str(exce.value).startswith("Expected 2 classes but got")


def test_if_run():
    """test if comparator was run before getting the best classifier"""
    cmp = BinaryClassifierComparator(X, y)
    with pytest.raises(AssertionError) as exce:
        cmp.get_best_clf()
    assert str(exce.value) == "There are no models to compare, you need to run the comparator first."
    with pytest.raises(AssertionError) as exce:
        cmp.get_metrics()
    assert str(exce.value) == "There are no metrics to be shown, you need to run the comparator first."


@pytest.mark.parametrize("metric_funcs, expected", [([f1_score, roc_auc_score], 3), ([roc_auc_score], 2)])
def test_number_of_metrics(metric_funcs, expected):
    """testing the number of metrics"""
    cmp = BinaryClassifierComparator(X, y, metric_funcs=metric_funcs)
    cmp.run()
    assert cmp.get_metrics().shape[1] == expected


def test_multiple_run_calls():
    """testing multiple calls of run()"""
    cmp = BinaryClassifierComparator(X, y)
    cmp.run()
    cmp.run()
    cmp.run()
    assert cmp.get_metrics().dropna().shape == (7,6)


####### Preprocess tests #######
@pytest.mark.parametrize(
    "classifiers, expected",
    [
        ({
            "pipe":Pipeline(
            steps=[("scaler", StandardScaler()), ("model", RandomForestClassifier(n_jobs=-1))])
        }, 1)
    ]
)
def test_pipelines_as_classifiers(classifiers, expected):
    """test the output of passing a single pipeline for classifiers"""
    cmp = BinaryClassifierComparator(X, y, classifiers=classifiers)
    cmp.run()
    assert cmp.get_metrics().shape[0] == expected


@pytest.mark.parametrize("preprocess, expected", [
    (Pipeline(steps=[("scaler", StandardScaler())]), (7,6))
])
def test_preprocess_attr_metrics(preprocess, expected):
    """testing the preprocess attribute"""
    cmp = BinaryClassifierComparator(X, y, preprocess=preprocess)
    cmp.run()
    assert cmp.get_metrics().shape == expected


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
    assert type(cmp.get_preprocess()) == Pipeline


def test_if_preprocess_pipeline():
    """testing if preprocess is an sklearn.pipeline.Pipeline"""
    preprocess = lambda x: x**2
    with pytest.raises(AssertionError) as exce:
        cmp = BinaryClassifierComparator(X, y, preprocess=preprocess)
    