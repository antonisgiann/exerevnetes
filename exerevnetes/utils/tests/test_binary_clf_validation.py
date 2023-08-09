import pytest
from exerevnetes.utils.binary_classification_validation import BinaryClassificationValidator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification


X, y = make_classification(n_samples=200, n_classes=2)

@pytest.mark.parametrize("estimators", [({"forest":RandomForestClassifier(), "extra":ExtraTreesClassifier()}), ({}), ([])])
def test_validate_classifiers(estimators):
    validator = BinaryClassificationValidator(estimators=estimators)
    if isinstance(estimators, dict) and len(estimators) != 0:
        assert validator.validate()
    elif isinstance(estimators, dict):
        with pytest.raises(ValueError) as exce:
            validator.validate()
    else:
        with pytest.raises(TypeError) as exce:
            validator.validate()