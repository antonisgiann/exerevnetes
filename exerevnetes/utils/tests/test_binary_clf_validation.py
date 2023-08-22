import pytest
from exerevnetes.utils.binary_classification_validation import BinaryClassificationValidator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


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


def test_validate_cv_and_exclude():
    validator = BinaryClassificationValidator(X, y, cv=5, exclude=["svc"])
    assert validator.validate()


@pytest.mark.parametrize(
        "estimators, cv, metric_funcs, preprocess, exclude, expected", [
            (
                {"Forest": RandomForestClassifier(), "extra":ExtraTreesClassifier()},
                3,
                [f1_score, roc_auc_score],
                Pipeline(steps=[("scaler", StandardScaler())]),
                None,
                "pass"
            ),
            (
                None,
                None,
                [f1_score],
                Pipeline(steps=[("scaler", StandardScaler())]),
                ["svc"],
                "pass"
            ),
            (
                None, None, lambda x: x**2, None, None, "fail"
            )
        ]
)
def test_validate_multiple_params(estimators, cv, metric_funcs, preprocess, exclude, expected):
    if expected == "pass":
        validator = BinaryClassificationValidator(
            X, y, 
            estimators=estimators,
            cv=cv, 
            metric_funcs=metric_funcs,
            preprocess=preprocess,
            exclude=exclude
            )
        assert validator.validate()
    if expected == "fail":
        validator = BinaryClassificationValidator(
            X, y,
            estimators=estimators,
            cv=cv,
            metric_funcs=metric_funcs,
            preprocess=preprocess,
            exclude=exclude
        )
        with pytest.raises(TypeError) as exce:
            validator.validate()