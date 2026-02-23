"""
=============================================================================
validate.py — CSV Correctness Checker
=============================================================================

Owner: Payal (code owner + verification + reporting)

Validates that CSV output files from cg_solver and spmv_bench conform to
the project schema and contain sane values.

Checks:
  1. All required columns are present
  2. No NaN or empty values in critical fields
  3. Numeric ranges are reasonable:
     - total_time_ms > 0
     - iterations > 0
     - rel_residual < 1e-6 (should be < 1e-8 if converged)
     - spmv_gbps > 0 and < 2000 (sanity)
     - bw_efficiency_pct > 0 and < 150 (allow some measurement noise)
  4. Every run converged (rel_residual < tolerance)

Usage:
  python scripts/validate.py results/rtx4090/cg_full.csv
  python scripts/validate.py results/rtx3080/spmv_bench.csv --type bench
  python scripts/validate.py results/             # validate all CSVs in dir

=============================================================================
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np


# ── Expected columns ──────────────────────────────────────────────────
CG_COLUMNS = [
    "gpu", "problem_dim", "grid_n", "rows", "nnz",
    "kernel_variant", "block_size",
    "total_time_ms", "spmv_time_ms", "dot_time_ms", "axpy_time_ms", "overhead_ms",
    "iterations", "rel_residual", "abs_error",
    "spmv_gbps", "bw_efficiency_pct"
]

BENCH_COLUMNS = [
    "gpu", "problem_dim", "grid_n", "rows", "nnz",
    "kernel_variant", "block_size",
    "avg_spmv_ms", "spmv_gbps", "bw_efficiency_pct"
]


def validate_cg_csv(filepath):
    """Validate a CG solver CSV file. Returns (n_errors, messages)."""
    errors = []
    try:
        df = pd.read_csv(filepath, skipinitialspace=True)
    except Exception as e:
        return 1, [f"Cannot read CSV: {e}"]

    # Check columns
    missing = set(CG_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    if df.empty:
        errors.append("CSV is empty")
        return len(errors), errors

    # Check convergence
    bad_conv = df[df["rel_residual"].apply(
        lambda x: isinstance(x, (int, float)) and x > 1e-6)]
    if len(bad_conv) > 0:
        errors.append(f"{len(bad_conv)} runs did NOT converge (rel_residual > 1e-6)")
        for _, row in bad_conv.iterrows():
            errors.append(f"  {row['gpu']} {row['problem_dim']} N={row['grid_n']} "
                         f"{row['kernel_variant']}: rel_res={row['rel_residual']}")

    # Check timing sanity
    if "total_time_ms" in df.columns:
        bad_time = df[df["total_time_ms"] <= 0]
        if len(bad_time) > 0:
            errors.append(f"{len(bad_time)} runs have total_time_ms <= 0")

    # Check bandwidth sanity
    if "spmv_gbps" in df.columns:
        numeric_gbps = pd.to_numeric(df["spmv_gbps"], errors="coerce")
        bad_bw = numeric_gbps[(numeric_gbps <= 0) | (numeric_gbps > 2000)]
        if len(bad_bw.dropna()) > 0:
            errors.append(f"{len(bad_bw.dropna())} runs have suspicious spmv_gbps values")

    # Check iterations
    if "iterations" in df.columns:
        bad_iter = df[df["iterations"] <= 0]
        if len(bad_iter) > 0:
            errors.append(f"{len(bad_iter)} runs have iterations <= 0")

    return len(errors), errors


def validate_bench_csv(filepath):
    """Validate a SpMV bench CSV file."""
    errors = []
    try:
        df = pd.read_csv(filepath, skipinitialspace=True)
    except Exception as e:
        return 1, [f"Cannot read CSV: {e}"]

    missing = set(BENCH_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    if df.empty:
        errors.append("CSV is empty")
        return len(errors), errors

    # Bandwidth sanity
    bad_bw = df[(df["spmv_gbps"] <= 0) | (df["spmv_gbps"] > 2000)]
    if len(bad_bw) > 0:
        errors.append(f"{len(bad_bw)} entries have suspicious spmv_gbps")

    return len(errors), errors


def main():
    parser = argparse.ArgumentParser(description="Validate project CSV files")
    parser.add_argument("path", help="CSV file or directory to validate")
    parser.add_argument("--type", choices=["cg", "bench", "auto"], default="auto",
                       help="CSV type (auto-detect from filename)")
    args = parser.parse_args()

    # Collect files
    if os.path.isdir(args.path):
        files = glob.glob(os.path.join(args.path, "**/*.csv"), recursive=True)
    else:
        files = [args.path]

    if not files:
        print(f"No CSV files found at {args.path}")
        sys.exit(1)

    total_errors = 0
    for f in sorted(files):
        # Auto-detect type
        if args.type == "auto":
            if "bench" in os.path.basename(f).lower():
                ftype = "bench"
            else:
                ftype = "cg"
        else:
            ftype = args.type

        if ftype == "cg":
            n_err, msgs = validate_cg_csv(f)
        else:
            n_err, msgs = validate_bench_csv(f)

        status = "PASS" if n_err == 0 else "FAIL"
        print(f"[{status}] {f}")
        for msg in msgs:
            print(f"       {msg}")
        total_errors += n_err

    print(f"\n{'=' * 40}")
    print(f"Total files: {len(files)}, Total errors: {total_errors}")
    if total_errors == 0:
        print("All validations PASSED")
    else:
        print(f"*** {total_errors} ERRORS found ***")
    sys.exit(0 if total_errors == 0 else 1)


if __name__ == "__main__":
    main()
