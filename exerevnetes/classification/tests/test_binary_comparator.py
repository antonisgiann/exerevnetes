import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from exerevnetes.classification import BinaryClassificationComparator

np.random.seed(47)

# General testing dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_classes=2, random_state=47)
X_df = pd.DataFrame(X)

X_df["category"] = 0.5
X_df["category"] = X_df["category"].map(lambda x: "A" if np.random.random() > x else "B").astype("category")
X_df.columns = X_df.columns.astype(str)


@pytest.mark.parametrize("X, y, estimators, cv, metric_funcs, prepro, expected_shape", [
    (
        X_df, # dependent variables
        y, # independent variables
        {"forest": RandomForestClassifier(), "extraTrees": ExtraTreesClassifier()}, # classifiers
        7, # cross validation folds
        [f1_score, roc_auc_score, accuracy_score], # results
        ColumnTransformer([
            ("numer", StandardScaler(), X_df.select_dtypes(include=["float64"]).columns.tolist()),
            ("categ", OneHotEncoder(), X_df.select_dtypes(include=["category"]).columns.tolist())
        ]), # preprocessing pipeline
        (2,4) # shape of the results output
    )
])
def test_general_class_functionality(X, y, estimators, cv, metric_funcs, prepro, expected_shape):
    """ Run a general test with most of the attributes set in the constructor """
    cmp = BinaryClassificationComparator(X, y, estimators, cv, metric_funcs, prepro)
    cmp.run()
    assert cmp.get_results(sort_by="f1_score").dropna().shape == expected_shape
    assert isinstance(cmp.get_params("preprocess")["preprocess"], (Pipeline, ColumnTransformer))


def test_default_params():
    """ Run a test with all default values for the comparator """
    cmp = BinaryClassificationComparator(X, y)
    cmp.run()
    assert cmp._results.dropna().shape == (7,6)
    m_f = cmp.get_params("metric_funcs")["metric_funcs"]
    assert isinstance(m_f, list) and len(m_f) == 5
    params =  cmp.get_params()
    assert params["exclude"] is None and params["preprocess"] is None and params["cv"] == 5
    assert len(params["estimators"]) == 7
    

def test_multiple_run_calls():
    """testing multiple calls of run()"""
    cmp = BinaryClassificationComparator(X, y)
    cmp.run()
    cmp.run()
    cmp.run()
    assert cmp._results.dropna().shape == (7,6)


@pytest.mark.parametrize("exclude", [(["random_forest", "random_forest"])])
def test_exclude_uniqueness(exclude):
    """test if exclude contains unique values"""
    with pytest.raises(AttributeError) as exce:
        cmp = BinaryClassificationComparator(X, y, exclude=exclude)


@pytest.mark.parametrize("estimators, expected", [
        ({"pipe":Pipeline(steps=[("scaler", StandardScaler()), ("model", RandomForestClassifier(n_jobs=-1))])}, 1)
    ])
def test_pipelines_as_classifiers(estimators, expected):
    """test the output of passing a single pipeline for classifiers"""
    cmp = BinaryClassificationComparator(X, y, estimators=estimators)
    cmp.run()
    assert cmp._results.dropna().shape[0] == expected


@pytest.mark.parametrize("preprocess", [
    (Pipeline(steps=[("scaler", StandardScaler()), ("model", RandomForestClassifier())]))
])
def test_if_preprocess_contains_predictor(preprocess):
    """testing if preprocessing contains a predictor"""
    with pytest.raises(AttributeError) as exce:
        cmp = BinaryClassificationComparator(X, y, preprocess=preprocess)


@pytest.mark.parametrize("exclude", [(["svc", "extra_trees"])])
def test_exlcude_setter(exclude):
    """test that you can exclude classifiers after initialization"""
    cmp = BinaryClassificationComparator(X, y, exclude=None)
    assert len(cmp.get_params("estimators")["estimators"]) == 7
    cmp.set_params(exclude=exclude)
    cmp.run()
    assert len(cmp.get_params("estimators")["estimators"]) == 5


def test_set_params_validation():
    """test setting wrong estimator object and wrong metric_funcs object"""
    cmp = BinaryClassificationComparator(X, y)
    with pytest.raises(TypeError) as exce:
        cmp.set_params(estimators=[])
    with pytest.raises(TypeError) as exce:
        cmp.set_params(metric_funcs={})
    with pytest.raises(TypeError) as exce:
        cmp.set_params(metric_funcs=[5, 7])


@pytest.mark.parametrize("metric", [("jiberish"), ("roc_auc_score")])
def test_get_best_clf(metric):
    """test getting best clf with wrong metric and with correct metric"""
    cmp = BinaryClassificationComparator(X, y)
    cmp.run()
    if metric == "jiberish":
        with pytest.raises(ValueError) as exce:
            cmp.get_best(metric=metric)
    else:
        cmp.get_best(metric=metric)