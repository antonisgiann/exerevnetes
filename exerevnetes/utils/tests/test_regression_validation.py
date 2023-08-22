import pytest
from exerevnetes.utils.regression_validation import RegressionValidator
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


X, y = make_regression(n_samples=200, n_features=10)


@pytest.mark.parametrize("estimators", [({"forest":RandomForestRegressor(), "extra":ExtraTreesRegressor()}), ({}), ([])])
def test_validate_regressors(estimators):
    validator = RegressionValidator(estimators=estimators)
    if isinstance(estimators, dict) and len(estimators) != 0:
        assert validator.validate()
    elif isinstance(estimators, dict):
        with pytest.raises(ValueError) as exce:
            validator.validate()
    else:
        with pytest.raises(TypeError) as exce:
            validator.validate()