from exerevnetes.base_validation import Validator

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