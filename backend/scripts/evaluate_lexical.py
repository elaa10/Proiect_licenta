
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
)

sys.path.insert(0, "/app")
from app.services.url_analyzer import extract_features, compute_lexical_score

RESULTS_DIR = "/app/results"


def load_dataset(csv_path: str, sample: int | None) -> pd.DataFrame:
    print(f"Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    url_col = next((c for c in df.columns if "url" in c), None)
    label_col = next((c for c in df.columns if c in ("label", "type", "class")), None)

    if not url_col or not label_col:
        raise ValueError(f"Cannot detect URL/label columns. Found: {list(df.columns)}")

    df = df[[url_col, label_col]].dropna()
    df.columns = ["url", "label"]

    label_map = {
        "bad": 1, "phishing": 1, "1": 1, 1: 1,
        "good": 0, "legitimate": 0, "benign": 0, "0": 0, 0: 0,
    }
    df["y"] = df["label"].map(label_map)
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)

    if sample and len(df) > sample:
        n = min(sample // 2, df["y"].sum(), (df["y"] == 0).sum())
        phishing = df[df["y"] == 1].sample(n, random_state=42)
        legitimate = df[df["y"] == 0].sample(n, random_state=42)
        df = pd.concat([phishing, legitimate]).sample(frac=1, random_state=42)
        print(f"Sampled {len(df)} URLs ({n} phishing + {n} legitimate)")

    print(f"Dataset: {len(df)} URLs | phishing: {df['y'].sum()} | legitimate: {(df['y']==0).sum()}")
    return df.reset_index(drop=True)


def compute_scores(df: pd.DataFrame) -> np.ndarray:
    print("Extracting features and computing scores...")
    scores = []
    for i, url in enumerate(df["url"]):
        if i % 10000 == 0 and i > 0:
            print(f"  {i}/{len(df)} processed...")
        try:
            scores.append(compute_lexical_score(extract_features(str(url))))
        except Exception:
            scores.append(0.0)
    return np.array(scores)


def metrics_at_threshold(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "fpr": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def find_best_f1_threshold(y_true, scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-9)
    mask = thresholds >= 0.05
    if mask.any():
        best_idx = np.argmax(f1s[:-1][mask])
        return float(thresholds[mask][best_idx])
    return float(thresholds[np.argmax(f1s[:-1])])


def print_metrics(m, label=""):
    tag = f" [{label}]" if label else ""
    print(f"\n  Threshold{tag}: {m['threshold']}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  FPR       : {m['fpr']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "lexical_evaluation.json"))
    args = parser.parse_args()

    df = load_dataset(args.csv, args.sample)
    scores = compute_scores(df)
    y = df["y"].values

    auc_roc = round(roc_auc_score(y, scores), 4)
    auc_pr = round(average_precision_score(y, scores), 4)
    optimal_tau = find_best_f1_threshold(y, scores)

    print("\n" + "="*60)
    print("LEXICAL DETECTOR — EVALUATION RESULTS")
    print("="*60)
    print(f"\n  AUC-ROC   : {auc_roc}")
    print(f"  AUC-PR    : {auc_pr}")
    print(f"  Optimal τ : {optimal_tau:.2f}")

    m_optimal = metrics_at_threshold(y, scores, optimal_tau)
    m_30 = metrics_at_threshold(y, scores, 0.30)
    m_40 = metrics_at_threshold(y, scores, 0.40)
    m_50 = metrics_at_threshold(y, scores, 0.50)

    print_metrics(m_optimal, "optimal τ")
    print_metrics(m_30, "τ=0.30")
    print_metrics(m_40, "τ=0.40")
    print_metrics(m_50, "τ=0.50")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    results = {
        "detector": "lexical_heuristic",
        "dataset_size": len(df),
        "phishing_count": int(y.sum()),
        "legitimate_count": int((y == 0).sum()),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "optimal_threshold": round(optimal_tau, 2),
        "metrics_optimal": m_optimal,
        "metrics_tau_030": m_30,
        "metrics_tau_040": m_40,
        "metrics_tau_050": m_50,
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {args.out}")
    print("="*60)


if __name__ == "__main__":
    main()