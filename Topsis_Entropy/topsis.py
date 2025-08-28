#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd

EPS = 1e-12
BENEFIT_COLS = ["Accuracy", "F1", "Recall", "AUC"]
COST_COLS = ["Train Time", "Inference Time"]

def entropy_weights(norm_matrix: np.ndarray) -> np.ndarray:
    """Entropy weights on the vector-normalized, cost-flipped matrix."""
    m, n = norm_matrix.shape
    col_sums = norm_matrix.sum(axis=0) + EPS
    P = norm_matrix / col_sums
    P = np.where(P <= 0, EPS, P)
    k = 1.0 / np.log(m)
    E = -k * np.sum(P * np.log(P), axis=0)
    d = 1.0 - E
    w = d / (d.sum() + EPS)
    return w

def topsis(data: pd.DataFrame, benefit_cols, cost_cols, auto_weights=True, weights=None):
    all_criteria = benefit_cols + cost_cols

    # --- 1) Vector normalization
    X = data[all_criteria].astype(float).values
    denom = np.sqrt((X ** 2).sum(axis=0)) + EPS
    R = X / denom  # r_ij

    # --- 2) Flip costs to benefits
    for j, col in enumerate(all_criteria):
        if col in cost_cols:
            R[:, j] = 1.0 - R[:, j]

    # --- 3) Weights
    if auto_weights:
        w = entropy_weights(R)
    else:
        if weights is None:
            w = np.ones(R.shape[1]) / R.shape[1]
        else:
            w = np.asarray(weights, dtype=float)
            w = w / (w.sum() + EPS)

    # --- 4) Weighted matrix and ideals (all treated as benefits now)
    V = R * w
    v_plus = V.max(axis=0)
    v_minus = V.min(axis=0)

    # --- 5) Distances & closeness scores
    d_plus = np.sqrt(((V - v_plus) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - v_minus) ** 2).sum(axis=1))
    cc = d_minus / (d_plus + d_minus + EPS)

    # --- 6) Prepare output frame + Weights row
    result = data.copy()
    result["Closeness Score"] = cc
    result = result.sort_values("Closeness Score", ascending=False)
    result["Rank"] = range(1, len(result) + 1)

    # restore index as "Alternative"
    result.reset_index(inplace=True)
    alt_name = data.index.name if data.index.name else "Embedding+Model"
    result.rename(columns={alt_name: "Alternative"}, inplace=True)

    # weights row
    w_row = {"Alternative": "Weights"}
    w_row.update({c: val for c, val in zip(all_criteria, w)})
    result = pd.concat([result, pd.DataFrame([w_row])], ignore_index=True)

    return result, w

def _validate_columns(df: pd.DataFrame):
    required = ["Accuracy", "F1", "Recall", "AUC", "Train Time", "Inference Time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _default_out_name(in_path: str, out_dir: str) -> str:
    base = os.path.splitext(os.path.basename(in_path))[0]
    return os.path.join(out_dir, f"{base}_topsis.csv")

def process_file(in_path: str, out_path: str):
    df = pd.read_csv(in_path, index_col=0)
    if df.index.name is None:
        df.index.name = "Embedding+Model"
    _validate_columns(df)

    result, _ = topsis(
        data=df,
        benefit_cols=BENEFIT_COLS,
        cost_cols=COST_COLS,
        auto_weights=True,   # per your usage
        weights=None
    )
    result.to_csv(out_path, index=False)
    print(f"Wrote: {os.path.abspath(out_path)}")

def main():
    parser = argparse.ArgumentParser(
        description="Run TOPSIS (auto entropy weights) over FOUR CSVs and write FOUR results."
    )
    parser.add_argument("inputs", nargs=4, help="Four input CSV paths")
    parser.add_argument("--outs", nargs=4, help="Four output CSV paths (optional)")
    parser.add_argument("--out_dir", default=".", help="Output directory if --outs not given")
    args = parser.parse_args()

    if args.outs and len(args.outs) != 4:
        parser.error("--outs must have exactly four paths when provided.")

    outs = args.outs or [ _default_out_name(p, args.out_dir) for p in args.inputs ]

    for in_path, out_path in zip(args.inputs, outs):
        process_file(in_path, out_path)

if __name__ == "__main__":
    main()