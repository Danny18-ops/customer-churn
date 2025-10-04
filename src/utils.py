import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str, target: str):
    df = pd.read_csv(path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    # Drop ID-like columns so the model never expects them at predict time
    id_like = {"customerid", "customer_id", "id"}
    drop_ids = [c for c in df.columns if c.lower().strip() in id_like]
    if drop_ids:
        df = df.drop(columns=drop_ids)

    y = df[target].astype(str)
    X = df.drop(columns=[target])
    return X, y

def infer_feature_types(df: pd.DataFrame):
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    num_cols = [c for c in df.columns if c not in cat_cols]
    return num_cols, cat_cols

def stratified_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
