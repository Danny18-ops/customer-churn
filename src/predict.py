import argparse
import pandas as pd
import joblib

def main(row_csv=None):
    model = joblib.load("models/model.joblib")
    if row_csv:
        X = pd.read_csv(row_csv)
    else:
        # example row (edit to match your columns)
        example = {
            "tenure": 12,
            "MonthlyCharges": 70.35,
            "TotalCharges": 845.4,
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check"
        }
        X = pd.DataFrame([example])

    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    for i, p in enumerate(pred):
        print(f"row {i}: churn={p}, prob={proba[i]:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--row_csv", help="CSV with rows to score (optional)")
    args = ap.parse_args()
    main(args.row_csv)

