from abc import ABC, abstractmethod


class Validator(ABC):
    def __init__(self, X=None, y=None, estimators=None, cv=None, metric_funcs=None, exclude=None):
        self.X = X
        self.y = y
        self.estimators = estimators
        self.cv = cv
        self.metric_funcs = metric_funcs
        self.exclude = exclude
    
    @classmethod
    def __validate_estimators_param(cls, estimators):
        if estimators != None and not isinstance(estimators, dict):
            raise TypeError("'estimators' must be a dictionary.")
        elif estimators != None and len(estimators) == 0:
            raise ValueError(f"'estimators' should not be an empty dictionary.")
        else:
            return True
        
    @classmethod
    def __validate_not_empty(cls, X, y):
        if X != None and y != None and (len(X) == 0 or len(y) == 0):
            raise ValueError("'X' and 'y' must not be empty arrays.")
        return True
        
    @classmethod
    def __validate_cv_param(cls, cv):
        if cv != None and isinstance(cv, int) and cv <= 1:
            raise ValueError("cv must be a positive integer greater than 1.")
        return True
        
    @classmethod
    def __validate_metric_funcs_param(cls, metric_funcs):
        if metric_funcs != None and not isinstance(metric_funcs, list) and not all(callable(m_f) for m_f in metric_funcs):
            raise TypeError("'metric_funcs' must be a list of callable functions.")
        elif metric_funcs != None and len(metric_funcs) == 0:
            raise ValueError(f"'metric_funcs' should not be an empty list.")
        else:
            return True

    @classmethod
    def __validate_X_y_mismatch(cls, X, y):
        if X != None and y != None and len(y) != len(X):
            raise ValueError(f"There is a mismatch between 'X' and 'y'. 'X' has shape {X.shape} and 'y' has shape {y.shape}")
        return True
    
    def run_base(self):
        self.__validate_estimators_param(self.estimators)
        self.__validate_not_empty(self.X, self.y)
        self.__validate_cv_param(self.cv)
        self.__validate_metric_funcs_param(self.metric_funcs)
        self.__validate_X_y_mismatch(self.X, self.y)

        return True
    
    @abstractmethod
    def validate(self):
        pass

        
        
        

        
    