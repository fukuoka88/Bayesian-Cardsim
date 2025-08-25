#!/usr/bin/env python3
"""
train.py — Cost-sensitive baseline for card/payments fraud, calibrated and thresholded for business costs.

Expected CSV schema (CardSim-like):
  txn_id, account_id, device_id, merchant_id, mcc, amount, timestamp, label
Notes:
  - timestamp should be parseable to UTC datetimes
  - label is {0,1} where 1 = fraud/chargeback
Usage:
  python train.py --train_csv data/cardsim_train.csv --test_csv data/cardsim_test.csv \
    --cost_fp 0.5 --cost_fn 6.0 --out_dir outputs/

This script will:
  1) Build pragmatic time, amount and simple velocity features
  2) Target-encode high-cardinality IDs (fit on train-only)
  3) Train a LightGBM model on a time-ordered train split
  4) Calibrate probabilities with isotonic regression on a holdout slice of train
  5) Choose a decision threshold that minimizes expected business cost
  6) Save metrics (JSON) and artifacts (model + calibrator via joblib)
"""
from __future__ import annotations

import argparse, os, math, json, warnings, pickle
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.isotonic import IsotonicRegression
from category_encoders.target_encoder import TargetEncoder
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

# ---------------------------- Feature engineering ----------------------------

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic numeric + temporal + simple per-account velocity features."""
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required")
    # Parse timestamps (UTC) and sort
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Amount transforms
    if "amount" not in df.columns:
        raise ValueError("amount column is required")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))

    # Time-of-day cyclic features
    df["hour"] = df["timestamp"].dt.hour
    df["dow"]  = df["timestamp"].dt.dayofweek
    df["sin_hour"] = np.sin(2 * math.pi * df["hour"]/24.0)
    df["cos_hour"] = np.cos(2 * math.pi * df["hour"]/24.0)

    # Simple expanding per-account velocity & spend profile
    if "account_id" in df.columns:
        df["acct_txn_count"] = df.groupby("account_id").cumcount()
        df["acct_mean_amt"]  = (
            df.groupby("account_id")["amount"]
              .expanding()
              .mean()
              .reset_index(level=0, drop=True)
              .fillna(0.0)
        )
    else:
        df["acct_txn_count"] = 0
        df["acct_mean_amt"]  = df["amount"]

    # Cast IDs to string/category to ensure encoders behave
    for col in ["account_id","device_id","merchant_id","mcc"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def fit_target_encoder(train_df: pd.DataFrame, cols: List[str], target_col: str):
    """Fit a smoothed target encoder on train-only and return fitted encoder."""
    if not cols:
        return None
    te = TargetEncoder(cols=cols, smoothing=10.0, min_samples_leaf=200)
    te.fit(train_df[cols], train_df[target_col])
    return te


def apply_target_encoder(te, df: pd.DataFrame) -> pd.DataFrame:
    """Apply target encoder producing te_* columns and preserving originals."""
    if te is None:
        return df
    te_df = te.transform(df[te.cols]).add_prefix("te_")
    return pd.concat([df.reset_index(drop=True), te_df.reset_index(drop=True)], axis=1)


# ---------------------------- Cost / threshold utils ----------------------------

def expected_cost(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, C_fp: float, C_fn: float) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return (C_fp * fp + C_fn * fn) / len(y_true)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, C_fp: float, C_fn: float,
                           grid: np.ndarray | None = None) -> Tuple[float, float]:
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    costs = [expected_cost(y_true, y_prob, t, C_fp, C_fn) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


# ---------------------------- Training pipeline ----------------------------

def train_eval(train_csv: str, test_csv: str, C_fp: float, C_fn: float, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    train_raw = pd.read_csv(train_csv)
    test_raw  = pd.read_csv(test_csv)

    # Basic checks
    if "label" not in train_raw.columns or "label" not in test_raw.columns:
        raise ValueError("Both train and test must include a 'label' column (0/1).")

    # Feature build
    train = make_features(train_raw)
    test  = make_features(test_raw)

    # Time-ordered split for calibration (last 10% of train for isotonic)
    split_idx = int(0.9 * len(train))
    tr_inner = train.iloc[:split_idx].copy()
    tr_hold  = train.iloc[split_idx:].copy()

    # Categorical id columns to encode
    cat_cols = [c for c in ["merchant_id","mcc","device_id","account_id"] if c in train.columns]
    num_cols = [c for c in ["log_amount","sin_hour","cos_hour","dow","acct_txn_count","acct_mean_amt","hour"] if c in train.columns]

    # Target encoder fit on inner-train only (no leakage)
    te = fit_target_encoder(tr_inner, cat_cols, target_col="label")

    tr_inner = apply_target_encoder(te, tr_inner)
    tr_hold  = apply_target_encoder(te, tr_hold)
    test_enc = apply_target_encoder(te, test)

    feat_cols = num_cols + [c for c in tr_inner.columns if c.startswith("te_")]
    X_tr, y_tr = tr_inner[feat_cols], tr_inner["label"].astype(int).values
    X_ho, y_ho = tr_hold[feat_cols], tr_hold["label"].astype(int).values
    X_te, y_te = test_enc[feat_cols], test_enc["label"].astype(int).values

    # LightGBM config (robust baseline)
    scale_pos_weight = max(1.0, (C_fn / max(C_fp, 1e-6)))
    params = dict(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=64,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=5.0,
        n_estimators=1500,
        random_state=RANDOM_STATE,
        class_weight={0: 1.0, 1: scale_pos_weight},
        n_jobs=-1,
    )

    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_tr, y_tr, eval_set=[(X_ho, y_ho)], eval_metric="auc", verbose=False)

    # Raw scores
    p_ho_raw = clf.predict_proba(X_ho)[:, 1]
    p_te_raw = clf.predict_proba(X_te)[:, 1]

    # Calibrate on holdout (isotonic)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_ho_raw, y_ho)
    p_te = iso.transform(p_te_raw)

    # Metrics
    roc  = roc_auc_score(y_te, p_te)
    pr   = average_precision_score(y_te, p_te)
    br   = brier_score_loss(y_te, p_te)
    tau, cost = find_optimal_threshold(y_te, p_te, C_fp, C_fn)

    y_hat = (p_te >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, y_hat, labels=[0,1]).ravel()

    report = {
        "ROC_AUC": float(roc),
        "PR_AUC": float(pr),
        "Brier": float(br),
        "Threshold": float(tau),
        "ExpectedCost_per_txn": float(cost),
        "Confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "FeaturesUsed": feat_cols,
        "TrainRows": int(len(train)),
        "TestRows": int(len(test)),
    }

    # Persist artifacts
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Save model + calibrator + encoder in one bundle
    bundle = {
        "model": clf,
        "calibrator": iso,
        "target_encoder": te,
        "feature_columns": feat_cols,
        "version": "1.0.0",
    }
    joblib.dump(bundle, os.path.join(out_dir, "model_bundle.joblib"))

    # Also export raw LightGBM text model (optional)
    try:
        clf.booster_.save_model(os.path.join(out_dir, "lightgbm.txt"))
    except Exception:
        pass

    return report


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="Path to training CSV (CardSim-like schema)")
    ap.add_argument("--test_csv", required=True, help="Path to test CSV (CardSim-like schema)")
    ap.add_argument("--cost_fp", type=float, default=0.5, help="Cost of falsely blocking a good txn")
    ap.add_argument("--cost_fn", type=float, default=6.0, help="Cost of missing a fraud txn")
    ap.add_argument("--out_dir", type=str, default="outputs", help="Where to save artifacts")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rpt = train_eval(args.train_csv, args.test_csv, args.cost_fp, args.cost_fn, args.out_dir)
    print(json.dumps(rpt, indent=2))
