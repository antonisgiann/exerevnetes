from exerevnetes.utils.validation import Validator
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB


default_estimators = {
    "random_forest": RandomForestClassifier(n_jobs=-1),
    "extra_trees": ExtraTreesClassifier(n_jobs=-1),
    "logistic_regression": LogisticRegression(),
    "svc": SVC(),
    "catboost": CatBoostClassifier(verbose=False, allow_writing_files=False),
    "xgboost": XGBClassifier(),
    "naive_bayes": GaussianNB()
}


class BinaryClassificationValidator(Validator):
    def __init__(self, X=None, y=None, estimators=None, cv=None, metric_funcs=None, exclude=None):
        super().__init__(X, y, estimators, cv, metric_funcs, exclude)
    
    @classmethod
    def __validate_binary_classification(cls, y):
        if y:
            n_classes = Counter(y)
            if len(n_classes) != 2:
                raise ValueError(f"Expected 2 classes but got {n_classes} classes.")
        return True
    
    @classmethod
    def __validate_exclude_param(cls, estimators, exclude):
        if estimators and exclude:
            raise AttributeError("You can't use the 'exclude' attribute when passing 'estimators'")
        elif exclude and (not isinstance(exclude, list) or not all(ex in default_estimators for ex in exclude)):
            raise ValueError(f"'exclude' must be a list of available default estimators. All default estimators are: {list(default_estimators.keys())}")
        elif exclude and (len(set(exclude)) != len(exclude)):
            raise ValueError("'exlucde' must contain unique estimators.")
        else:
            return True
        
    def validate(self):
        base_val = self.run_base_validation()
        if self.y != None:
            base_val = base_val and self.__validate_binary_classification(self.y) 
        if self.exclude != None:
            base_val = base_val and self.__validate_exclude_param(self.estimators, self.exclude)
        return base_val
        