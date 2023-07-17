import time
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score

from exerevnetes.utils import time_format


default_classifiers = {
    "random_forest": RandomForestClassifier(),
    "extra_tree": ExtraTreesClassifier(),
    "logistic_reg": LogisticRegression(),
    "svc": SVC()
}

metric_func = [
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    accuracy_score
]

class BinaryClassifierComparator:
    def __init__(self, X, y, classifiers: dict = default_classifiers, cv=10):
        self.X = X
        self.y = y
        self.classifiers = classifiers
        self.cv = cv
        self.metrics = {}
    
    def run(self):
        print("The comparator has started...")
        initial_time = time.time()
        for clf_name, clf in self.classifiers.items():
            print(f"Running cross validation for {clf_name}...")
            clf_time = time.time()
            preds = cross_val_predict(clf, self.X, self.y, cv=self.cv)
            cv_time = time.time() - clf_time
            self.metrics[clf_name] = {"cv_time": cv_time}
            self.calculate_scores(clf_name, preds)
        print(f"Total comparator time {time_format(time.time() - initial_time)}")
        self.format_metrics()

    def calculate_scores(self, clf_name, preds):
        for m in metric_func:
            self.metrics[clf_name][m.__name__] = m(self.y, preds)

    def format_metrics(self):
        assert(len(self.metrics) != 0), "You need to run the comparator first"
        self.metrics = pd.DataFrame(self.metrics).T

    def best_clf(self, metric="f1_score"):
        assert(len(self.metrics) != 0), "You need to run the comparator first"
        return self.classifiers[self.metrics.sort_values(by=metric).index[-1]]