import pytest
from exerevnetes.regression import RegressionComparator

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression


X, y = make_regression(500, 10, random_state=47)


@pytest.mark.parametrize("X, y, estimators, cv, metric_funcs, prepro, expected_shape", [
    (
        X,
        y, 
        {"l2": Ridge(), "elastic": ElasticNet(), "forest": RandomForestRegressor()},
        3,
        [mean_absolute_error],
        Pipeline(steps=[("scale", StandardScaler())]),
        (3,2)
    )
])
def test_general_class_functionality(X, y, estimators, cv, metric_funcs, prepro, expected_shape):
    """ run a general test with most of attributes set"""
    cmp = RegressionComparator(X, y, estimators, cv, metric_funcs, prepro)
    cmp.run()
    assert cmp._results.shape == expected_shape