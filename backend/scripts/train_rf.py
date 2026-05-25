"""
Random Forest training si evaluare pentru detectia URL-urilor de phishing.

Modificari fata de versiunea anterioara:
  - Dataset principal: PhiUSIIL (UCI, 2024) — URL-urile au scheme http(s)://
    si reprezinta distributia reala a inputului utilizatorului.
  - Label map invers: in PhiUSIIL, label=1 = legitimate, label=0 = phishing.
  - Evaluare OOD optionala pe GramBeddings sau Kaggle vechi.

Produce:
    - /app/models/rf_model.joblib     modelul antrenat (final)
    - /app/results/rf_evaluation.json metrici complete pentru Cap. 4.5

Utilizare (in containerul backend):
    MSYS_NO_PATHCONV=1 docker exec -it proiect_licenta-backend-1 \\
        python scripts/train_rf.py \\
        --csv /app/data/PhiUSIIL_Phishing_URL_Dataset.csv \\
        --sample 150000
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, "/app")
from app.services.url_analyzer import extract_features

MODELS_DIR = "/app/models"
RESULTS_DIR = "/app/results"

FEATURE_ORDER = [
    "url_length", "hostname_length", "path_length",
    "num_dots", "num_hyphens", "num_slashes", "num_underscores", "num_question_marks",
    "has_at_symbol", "num_subdomains", "has_ip_address", "is_https",
    "is_url_shortener", "is_punycode", "suspicious_keyword_count",
    "digit_ratio", "has_suspicious_tld", "double_slash_in_path",
    "min_brand_levenshtein", "sld_is_exact_brand",
]


def load_dataset(csv_path: str, sample: int | None) -> pd.DataFrame:
    """
    Incarca un dataset de URL-uri si normalizeaza coloanele la (url, y).

    Detecteaza automat schema labelului:
      - Coloana 'y' (sau 'Y'): format intern, y=1 = phishing. Nu reinterpreteaza.
      - PhiUSIIL: label=1 -> legitimate (y=0), label=0 -> phishing (y=1)
      - Kaggle:   Label='bad' -> phishing (y=1), Label='good' -> legitimate (y=0)

    Pentru proiectul nostru, conventia interna este y=1 pentru phishing.
    """
    print(f"[load] Reading {csv_path} ...")
    df = pd.read_csv(csv_path)

    cols_lower = {c.lower(): c for c in df.columns}
    url_col = cols_lower.get("url")
    y_col = cols_lower.get("y")  # format intern (din prepare_dataset.py)
    label_col = cols_lower.get("label") or cols_lower.get("type") or cols_lower.get("class")

    if not url_col:
        raise ValueError(f"Cannot detect URL column. Found: {list(df.columns)}")

    if y_col is not None:
        # Format intern: y=1 = phishing. Folosim direct, fara reinterpretare.
        print("[load] Detected internal schema (column 'y'): y=1 means phishing")
        df = df[[url_col, y_col]].dropna()
        df.columns = ["url", "y"]
        df["y"] = df["y"].astype(int)
    elif label_col is not None:
        df = df[[url_col, label_col]].dropna()
        df.columns = ["url", "label"]
        label_values = set(df["label"].unique())
        is_phiusiil_schema = label_values.issubset({0, 1, "0", "1"})
        if is_phiusiil_schema:
            print("[load] Detected PhiUSIIL schema: label=1 means legitimate")
            df["y"] = (df["label"].astype(int) == 0).astype(int)
        else:
            print("[load] Detected Kaggle schema: 'bad' means phishing")
            label_map = {"bad": 1, "phishing": 1, "good": 0, "legitimate": 0, "benign": 0}
            df["y"] = df["label"].astype(str).str.lower().map(label_map)
            df = df.dropna(subset=["y"])
            df["y"] = df["y"].astype(int)
        df = df.drop(columns=["label"])
    else:
        raise ValueError(f"Cannot detect label column. Found: {list(df.columns)}")

    print(f"[load] Total: {len(df)}  phishing={int(df['y'].sum())}  legitimate={int((df['y']==0).sum())}")

    if sample and len(df) > sample:
        n = min(sample // 2, int(df["y"].sum()), int((df["y"] == 0).sum()))
        phishing = df[df["y"] == 1].sample(n, random_state=42)
        legitimate = df[df["y"] == 0].sample(n, random_state=42)
        df = pd.concat([phishing, legitimate]).sample(frac=1, random_state=42)
        print(f"[load] Balanced sample: {len(df)} URLs ({n} phishing + {n} legitimate)")

    return df.reset_index(drop=True)


def build_feature_matrix(urls: pd.Series) -> np.ndarray:
    print(f"[features] Extracting 20 features from {len(urls)} URLs ...")
    rows = []
    for i, url in enumerate(urls):
        if i and i % 20000 == 0:
            print(f"[features]   {i}/{len(urls)}")
        try:
            f = extract_features(str(url))
            rows.append([f[k] for k in FEATURE_ORDER])
        except Exception:
            rows.append([0] * len(FEATURE_ORDER))
    return np.asarray(rows, dtype=np.float32)


def metrics_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "fpr": round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def find_best_f1_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions[:-1] * recalls[:-1] / np.maximum(precisions[:-1] + recalls[:-1], 1e-9)
    return float(thresholds[int(np.argmax(f1s))])


def cross_validate(X, y, n_splits=5, n_estimators=200, random_state=42):
    print(f"[cv] Stratified {n_splits}-fold cross-validation ...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[val_idx])[:, 1]
        m = {
            "fold": fold,
            "auc_roc": round(roc_auc_score(y[val_idx], proba), 4),
            "auc_pr": round(average_precision_score(y[val_idx], proba), 4),
            "f1_at_05": round(f1_score(y[val_idx], (proba >= 0.5).astype(int)), 4),
        }
        fold_metrics.append(m)
        print(f"[cv]   fold {fold}: AUC-ROC={m['auc_roc']}  AUC-PR={m['auc_pr']}  F1@0.5={m['f1_at_05']}")

    summary = {
        "auc_roc_mean": round(float(np.mean([m["auc_roc"] for m in fold_metrics])), 4),
        "auc_roc_std":  round(float(np.std([m["auc_roc"] for m in fold_metrics])), 4),
        "auc_pr_mean":  round(float(np.mean([m["auc_pr"]  for m in fold_metrics])), 4),
        "f1_mean":      round(float(np.mean([m["f1_at_05"] for m in fold_metrics])), 4),
    }
    return fold_metrics, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/app/data/PhiUSIIL_Phishing_URL_Dataset.csv")
    parser.add_argument("--sample", type=int, default=150000)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--model-out", default=os.path.join(MODELS_DIR, "rf_model.joblib"))
    parser.add_argument("--results-out", default=os.path.join(RESULTS_DIR, "rf_evaluation.json"))
    parser.add_argument("--oob-csv", default=None,
                        help="Optional CSV pentru evaluare out-of-distribution (e.g. dataset Kaggle vechi)")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.perf_counter()
    df = load_dataset(args.csv, args.sample)
    X = build_feature_matrix(df["url"])
    y = df["y"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[split] train={len(X_train)}  test={len(X_test)}")

    fold_metrics, cv_summary = cross_validate(
        X_train, y_train, n_splits=5, n_estimators=args.n_estimators
    )

    print(f"[train] Fitting final model on full training set ({len(X_train)} samples) ...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    auc_roc = round(roc_auc_score(y_test, proba_test), 4)
    auc_pr  = round(average_precision_score(y_test, proba_test), 4)
    best_tau = find_best_f1_threshold(y_test, proba_test)

    m_best = metrics_at_threshold(y_test, proba_test, best_tau)
    m_05   = metrics_at_threshold(y_test, proba_test, 0.5)
    m_03   = metrics_at_threshold(y_test, proba_test, 0.3)
    m_07   = metrics_at_threshold(y_test, proba_test, 0.7)

    importances = sorted(
        zip(FEATURE_ORDER, clf.feature_importances_.tolist()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    importances = [{"feature": k, "importance": round(v, 4)} for k, v in importances]

    elapsed = round(time.perf_counter() - t0, 1)

    print("\n" + "=" * 60)
    print("RANDOM FOREST — EVALUATION RESULTS (held-out test set)")
    print("=" * 60)
    print(f"\n  AUC-ROC      : {auc_roc}")
    print(f"  AUC-PR       : {auc_pr}")
    print(f"  Optimal F1 tau: {best_tau:.4f}")
    print(f"  Best F1      : {m_best['f1']}  (precision={m_best['precision']}, recall={m_best['recall']})")
    print(f"\n  Top 5 features by importance:")
    for f in importances[:5]:
        print(f"    {f['feature']:<28} {f['importance']}")
    print(f"\n  Total time: {elapsed}s")
    print("=" * 60)

    joblib.dump(
        {"model": clf, "feature_order": FEATURE_ORDER, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        args.model_out,
    )
    print(f"[save] Model: {args.model_out}")

    results = {
        "classifier": "RandomForest",
        "training_dataset": os.path.basename(args.csv),
        "n_estimators": args.n_estimators,
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "phishing_count": int(y.sum()),
        "legitimate_count": int((y == 0).sum()),
        "cross_validation": {"folds": fold_metrics, "summary": cv_summary},
        "test_set": {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "optimal_threshold_f1": round(best_tau, 4),
            "metrics_optimal_f1": m_best,
            "metrics_tau_030": m_03,
            "metrics_tau_050": m_05,
            "metrics_tau_070": m_07,
        },
        "feature_importance": importances,
        "elapsed_seconds": elapsed,
    }

    # Evaluare optionala out-of-distribution
    if args.oob_csv and os.path.exists(args.oob_csv):
        print(f"\n[oob] Evaluating on out-of-distribution dataset: {args.oob_csv}")
        try:
            df_oob = load_dataset(args.oob_csv, sample=50000)
            X_oob = build_feature_matrix(df_oob["url"])
            y_oob = df_oob["y"].to_numpy()
            proba_oob = clf.predict_proba(X_oob)[:, 1]
            oob_results = {
                "dataset": os.path.basename(args.oob_csv),
                "size": len(df_oob),
                "auc_roc": round(roc_auc_score(y_oob, proba_oob), 4),
                "auc_pr":  round(average_precision_score(y_oob, proba_oob), 4),
                "metrics_tau_050": metrics_at_threshold(y_oob, proba_oob, 0.5),
            }
            print(f"[oob]  AUC-ROC: {oob_results['auc_roc']}  AUC-PR: {oob_results['auc_pr']}")
            results["oob_evaluation"] = oob_results
        except Exception as e:
            print(f"[oob]  Failed: {e}")

    with open(args.results_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] Results: {args.results_out}")


if __name__ == "__main__":
    main()