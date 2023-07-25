import pytest
import numpy as np

from .._clf_comparator import BinaryClassifierComparator
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, roc_auc_score


np.random.seed(47)

# General testing dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=47)


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
        cmp.best_clf()
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