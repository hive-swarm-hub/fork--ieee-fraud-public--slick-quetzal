"""
Train Lasso on Chronon-backfilled features and report AUC-PR.

This script runs ON the Chronon server (Railway).
It reads the backfilled join CSV and trains a Lasso model.

Do not modify.
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SPLIT_DT = 11246620  # 75th percentile of TransactionDT


def main():
    t0 = time.time()

    # Read backfilled features
    print("Loading backfilled features...")
    df = pd.read_csv("/tmp/fraud_features_backfill.csv")
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # Extract target and split column
    if "isFraud" not in df.columns:
        # Try alternate names from Chronon output
        for col in df.columns:
            if "isfraud" in col.lower():
                df["isFraud"] = df[col]
                break
    if "TransactionDT" not in df.columns:
        for col in df.columns:
            if "transactiondt" in col.lower():
                df["TransactionDT"] = df[col]
                break

    y = df["isFraud"].values.astype(np.float64)
    transaction_dt = df["TransactionDT"].values
    print(f"  Fraud rate: {y.mean():.4f}")

    # Drop non-feature columns
    drop_cols = [c for c in df.columns if c.lower() in (
        "isfraud", "transactionid", "transactiondt", "ts", "ds",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "productcd", "addr1", "addr2", "p_emaildomain", "r_emaildomain",
    )]
    # Also drop string columns
    feature_df = df.drop(columns=drop_cols, errors="ignore")
    string_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    feature_df = feature_df.drop(columns=string_cols, errors="ignore")

    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)
    print(f"  Feature columns: {len(feature_names)}")

    # Time-based split
    train_mask = transaction_dt <= SPLIT_DT
    test_mask = transaction_dt > SPLIT_DT
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"  Train: {X_train.shape[0]} rows ({y_train.mean():.4f} fraud rate)")
    print(f"  Test:  {X_test.shape[0]} rows ({y_test.mean():.4f} fraud rate)")

    if X_test.shape[0] == 0:
        print("ERROR: No test rows. Check TransactionDT split cutoff.")
        print("---")
        print("auc_pr:           0.000000")
        print("n_features:       0")
        print("n_nonzero:        0")
        print("train_auc_pr:     0.000000")
        return

    # Handle infinities and NaNs
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # Impute
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Lasso
    print("Training Lasso...")
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Score
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    train_auc_pr = average_precision_score(y_train, y_train_prob)
    test_auc_pr = average_precision_score(y_test, y_test_prob)

    n_features = X.shape[1]
    n_nonzero = int(np.sum(model.coef_ != 0))
    elapsed = time.time() - t0

    # Top features
    coef_abs = np.abs(model.coef_[0])
    top_idx = np.argsort(coef_abs)[::-1][:10]
    print("\nTop features by |coefficient|:")
    for i, idx in enumerate(top_idx):
        if coef_abs[idx] > 0:
            print(f"  {i+1}. {feature_names[idx]}: {model.coef_[0][idx]:.4f}")

    print(f"\nElapsed: {elapsed:.1f}s")
    print("---")
    print(f"auc_pr:           {test_auc_pr:.6f}")
    print(f"n_features:       {n_features}")
    print(f"n_nonzero:        {n_nonzero}")
    print(f"train_auc_pr:     {train_auc_pr:.6f}")


if __name__ == "__main__":
    main()
