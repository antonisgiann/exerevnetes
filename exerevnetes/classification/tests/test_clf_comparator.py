import pytest
import numpy as np

from .._clf_comparator import BinaryClassifierComparator


np.random.seed(47)


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
    assert cmp.metrics.shape == expected


@pytest.mark.parametrize(
    "X, y",
    [(np.random.randint(0, 100, size=1000).reshape(100,10), np.random.randint(0, 1, size=100))]
)
def test_single_class(X, y):
    """test if there is less than two classes in y_true."""
    cmp = BinaryClassifierComparator(X, y)
    with pytest.raises(ValueError) as exce:
        cmp.run()
    assert str(exce.value) == "Expected 2 classes but got less in y_true"