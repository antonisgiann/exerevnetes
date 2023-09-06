from exerevnetes.base_validation import Validator
from collections import Counter

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


default_estimators = {
    "linear": LinearRegression(),
    "ridge": Ridge(),
    "forest": RandomForestRegressor(n_jobs=-1),
    "extra_trees": ExtraTreesRegressor(n_jobs=-1),
    "svr": SVR(),
    "xgboost": XGBRegressor(),
    "catboost": CatBoostRegressor(verbose=False, allow_writing_files=False),
}

class RegressionValidator(Validator):
    def __init__(self, X=None, y=None, estimators=None, cv=None, metric_funcs=None, preprocess=None, exclude=None):
        super().__init__(X, y, estimators, cv, metric_funcs, preprocess, exclude)

    @classmethod
    def __validate_regression(cls, y):
        if y is not None:
            n_classes = Counter(y)
            if len(n_classes) <= 100 or isinstance(y, float):
                raise ValueError(f"`y` looks like a discrete variable.`")
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
        if self.y is not None:
            base_val = base_val and self.__validate_regression(self.y) 
        if self.exclude is not None:
            base_val = base_val and self.__validate_exclude_param(self.estimators, self.exclude)
        return base_val