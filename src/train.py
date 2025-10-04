import argparse
import os
import json
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from utils import load_data, infer_feature_types, stratified_split

def build_pipeline(num_cols, cat_cols):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols)
    ])
    model = LogisticRegression(max_iter=200)
    return Pipeline([("preprocess", pre), ("clf", model)])

def main(data_path, target):
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    X, y = load_data(data_path, target)
    num_cols, cat_cols = infer_feature_types(X)
    pipe = build_pipeline(num_cols, cat_cols)

    grid = {
        "clf__C": [0.1, 1.0, 3.0],
        "clf__solver": ["lbfgs"]
    }

    X_train, X_test, y_train, y_test = stratified_split(X, y)

    gs = GridSearchCV(pipe, grid, cv=5, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    preds = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]

    metrics = {
        "best_params": gs.best_params_,
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "report": classification_report(y_test, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }

    joblib.dump(best, "models/model.joblib")
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Best params:", gs.best_params_)
    print("ROC-AUC:", metrics["roc_auc"])
    print("Confusion matrix:", metrics["confusion_matrix"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/telco_churn.csv")
    ap.add_argument("--target", default="Churn")
    args = ap.parse_args()
    main(args.data, args.target)
