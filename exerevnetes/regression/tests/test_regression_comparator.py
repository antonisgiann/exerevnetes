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
    assert cmp.results_.shape == expected_shape


@pytest.mark.parametrize("estimators", [([]), ({})])
def test_wrong_estimators(estimators):
    with pytest.raises(Exception):
        cmp = RegressionComparator(X, y, estimators=estimators)


def test_default_params():
    """ Run a test with all default values for the comparator """
    cmp = RegressionComparator(X, y)
    cmp.run()
    assert cmp.results_.dropna().shape == (7,4)
    m_f = cmp.get_params("metric_funcs")["metric_funcs"]
    assert isinstance(m_f, list) and len(m_f) == 3
    params =  cmp.get_params()
    assert params["exclude"] is None and params["preprocess"] is None and params["cv"] == 5
    assert len(params["estimators"]) == 7


@pytest.mark.parametrize("estimators", [({"forest": RandomForestRegressor(), "Ridge": Ridge()})])
def test_multiple_run_calls(estimators):
    cmp = RegressionComparator(X, y, estimators=estimators)
    cmp.run()
    cmp.run()
    cmp.run()
    assert cmp.results_.dropna().shape == (2,4)


@pytest.mark.parametrize("exclude", [(["random_forest", "random_forest"])])
def test_exclude_uniqueness(exclude):
    """test if exclude contains unique values"""
    with pytest.raises(AttributeError) as exce:
        cmp = RegressionComparator(X, y, exclude=exclude)


@pytest.mark.parametrize("estimators, expected", [
        ({"pipe":Pipeline(steps=[("scaler", StandardScaler()), ("model", RandomForestRegressor(n_jobs=-1))])}, 1)
    ])
def test_pipelines_as_classifiers(estimators, expected):
    """test the output of passing a single pipeline for classifiers"""
    cmp = RegressionComparator(X, y, estimators=estimators)
    cmp.run()
    assert cmp.results_.dropna().shape[0] == expected


@pytest.mark.parametrize("preprocess", [
    (Pipeline(steps=[("scaler", StandardScaler()), ("model", RandomForestRegressor())]))
])
def test_if_preprocess_contains_predictor(preprocess):
    """testing if preprocessing contains a predictor"""
    with pytest.raises(AttributeError) as exce:
        cmp = RegressionComparator(X, y, preprocess=preprocess)


@pytest.mark.parametrize("exclude", [(["svr", "extra_trees"])])
def test_exlcude_setter(exclude):
    """test that you can exclude classifiers after initialization"""
    cmp = RegressionComparator(X, y, exclude=None)
    assert len(cmp.get_params("estimators")["estimators"]) == 7
    cmp.set_params(exclude=exclude)
    cmp.run()
    assert len(cmp.get_params("estimators")["estimators"]) == 5


def test_set_params_validation():
    """test setting wrong estimator object and wrong metric_funcs object"""
    cmp = RegressionComparator(X, y)
    with pytest.raises(TypeError) as exce:
        cmp.set_params(estimators=[])
    with pytest.raises(TypeError) as exce:
        cmp.set_params(metric_funcs={})
    with pytest.raises(TypeError) as exce:
        cmp.set_params(metric_funcs=[5, 7])


@pytest.mark.parametrize("metric", [("jiberish"), ("mean_absolute_percentage_error")])
def test_get_best_estimator(metric):
    """test getting best clf with wrong metric and with correct metric"""
    cmp = RegressionComparator(X, y)
    cmp.run()
    if metric == "jiberish":
        with pytest.raises(ValueError) as exce:
            cmp.get_best(metric=metric)
    else:
        cmp.get_best(metric=metric)