from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_model(name):
    if name == "logistic":
        return LogisticRegression(max_iter=1000)
    if name == "svm":
        return SVC(probability=True)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200)
    if name == "xgb":
        return XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False
        )
    raise ValueError("Unknown model")
