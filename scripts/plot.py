"""
=============================================================================
plot.py — Generate All Project Charts from CSV Data
=============================================================================

Owner: Payal (code owner + verification + reporting)
Recent edit on this file: Avah (refactoring of file import/export)

Reads the standardized CSV files produced by cg_solver and spmv_bench
on each GPU, then generates the 8 deliverable charts from the project plan:

  1. Time-to-solution vs. problem size (2D + 3D)
  2. Iteration time breakdown (stacked bars)
  3. SpMV bandwidth (GB/s) vs. problem size
  4. Bandwidth efficiency (%) vs. problem size
  5. Kernel variant comparison (grouped bars)
  6. Block size sweep (throughput vs block size)
  7. 4090 vs 3080 speedup ratio
  8. Roofline placeholder (annotated from Nsight screenshots)

Usage:
  python scripts/plot.py                                   # default paths
  python scripts/plot.py --dir results/ --out report/figures/

Input CSV schema (cg_solver output):
  gpu, problem_dim, grid_n, rows, nnz,
  kernel_variant, block_size,
  total_time_ms, spmv_time_ms, dot_time_ms, axpy_time_ms, overhead_ms,
  iterations, rel_residual, abs_error,
  spmv_gbps, bw_efficiency_pct

Input CSV schema (spmv_bench output):
  gpu, problem_dim, grid_n, rows, nnz,
  kernel_variant, block_size,
  avg_spmv_ms, spmv_gbps, bw_efficiency_pct

Requirements:
  pip install matplotlib pandas seaborn

=============================================================================
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Plot style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {
    "RTX 4090":       "#76B900",   # NVIDIA green
    "RTX 3080":       "#1E88E5",   # blue
    "MX250":          "#FF7043",   # orange
    "cusparse":       "#424242",   # dark gray
    "row_per_thread": "#1E88E5",   # blue
    "warp_per_row":   "#FF7043",   # orange
}
FIG_DPI = 150

def load_csv(filepath):
    """Try multiple encodings and return a DataFrame or None."""
    for encoding in ["utf-8-sig", "utf-16", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(filepath, skipinitialspace=True, encoding=encoding)
            print(f"  Loaded {filepath} ({encoding}): {len(df)} rows")
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    print(f"  ERROR: Could not load {filepath} with any known encoding")
    return None


def load_cg_csvs(results_dir):
    files = glob.glob(os.path.join(results_dir, "cg_*.csv")) + \
            glob.glob(os.path.join(results_dir, "*/cg_*.csv"))
    if not files:
        print(f"WARNING: No CG CSV files found in {results_dir}")
        return pd.DataFrame()

    dfs = [df for f in files if (df := load_csv(f)) is not None]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_bench_csvs(results_dir):
    files = glob.glob(os.path.join(results_dir, "spmv_*.csv")) + \
            glob.glob(os.path.join(results_dir, "*/spmv_*.csv"))
    if not files:
        print(f"WARNING: No SpMV bench CSVs found in {results_dir}")
        return pd.DataFrame()

    dfs = [df for f in files if (df := load_csv(f)) is not None]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
#  Chart 1: Time-to-Solution vs Problem Size
# ═══════════════════════════════════════════════════════════════════════
def plot_time_vs_size(df, out_dir, dim="2D"):
    """
    X-axis: grid_n (problem size)
    Y-axis: total_time_ms
    One line per GPU, using cuSPARSE baseline for clean comparison.
    """
    subset = df[(df["problem_dim"] == dim) & (df["kernel_variant"] == "cusparse")]
    if subset.empty:
        print(f"  Skip chart 1 ({dim}): no cusparse data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for gpu_name, grp in subset.groupby("gpu"):
        grp_sorted = grp.sort_values("grid_n")
        color = COLORS.get(gpu_name, "#333333")
        ax.plot(grp_sorted["grid_n"], grp_sorted["total_time_ms"],
                marker="o", label=gpu_name, color=color, linewidth=2)

    ax.set_xlabel("Grid Size N")
    ax.set_ylabel("Time to Solution (ms)")
    ax.set_title(f"CG Time-to-Solution — {dim} Poisson (cuSPARSE baseline)")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"chart1_time_vs_size_{dim}.png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  ✓ Chart 1 ({dim}): time_vs_size")


# ═══════════════════════════════════════════════════════════════════════
#  Chart 2: Iteration Time Breakdown (Stacked Bars)
# ═══════════════════════════════════════════════════════════════════════
def plot_time_breakdown(df, out_dir):
    """
    Stacked bar chart showing SpMV / dot / axpy / overhead per config.
    One group per GPU × problem size, cusparse baseline.
    """
    subset = df[df["kernel_variant"] == "cusparse"].copy()
    if subset.empty:
        print("  Skip chart 2: no cusparse data")
        return

    # Use largest 2D problem per GPU for a clean chart
    subset = subset[subset["problem_dim"] == "2D"]
    idx = subset.groupby("gpu")["grid_n"].idxmax()
    subset = subset.loc[idx].sort_values("gpu")

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = subset["gpu"] + " (N=" + subset["grid_n"].astype(str) + ")"
    x = range(len(labels))
    width = 0.5

    # Convert to per-iteration averages
    for col in ["spmv_time_ms", "dot_time_ms", "axpy_time_ms", "overhead_ms"]:
        subset[col + "_per_iter"] = subset[col] / subset["iterations"]

    bottoms = [0] * len(subset)
    phase_colors = {"spmv": "#D32F2F", "dot": "#1976D2", "axpy": "#388E3C", "overhead": "#FFA000"}
    for phase, color in phase_colors.items():
        vals = subset[f"{phase}_time_ms_per_iter"].values
        ax.bar(x, vals, width, bottom=bottoms, label=phase, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Time per Iteration (ms)")
    ax.set_title("CG Iteration Time Breakdown — 2D Poisson (cuSPARSE)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart2_time_breakdown.png"), dpi=FIG_DPI)
    plt.close(fig)
    print("  ✓ Chart 2: time_breakdown")


# ═══════════════════════════════════════════════════════════════════════
#  Chart 3: SpMV Bandwidth vs Problem Size
# ═══════════════════════════════════════════════════════════════════════
def plot_bandwidth_vs_size(df, out_dir, dim="2D"):
    """SpMV achieved GB/s vs matrix rows, one line per GPU (cusparse)."""
    subset = df[(df["problem_dim"] == dim) & (df["kernel_variant"] == "cusparse")]
    if subset.empty:
        print(f"  Skip chart 3 ({dim}): no data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for gpu_name, grp in subset.groupby("gpu"):
        grp_sorted = grp.sort_values("rows")
        ax.plot(grp_sorted["rows"], grp_sorted["spmv_gbps"],
                marker="s", label=gpu_name,
                color=COLORS.get(gpu_name, "#333"), linewidth=2)

    ax.set_xlabel("Matrix Rows")
    ax.set_ylabel("SpMV Achieved Bandwidth (GB/s)")
    ax.set_title(f"SpMV Bandwidth — {dim} Poisson (cuSPARSE)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"chart3_bandwidth_vs_size_{dim}.png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  ✓ Chart 3 ({dim}): bandwidth_vs_size")


# ═══════════════════════════════════════════════════════════════════════
#  Chart 4: Bandwidth Efficiency (%)
# ═══════════════════════════════════════════════════════════════════════
def plot_bw_efficiency(df, out_dir, dim="2D"):
    """% of peak device bandwidth achieved, per GPU."""
    subset = df[(df["problem_dim"] == dim) & (df["kernel_variant"] == "cusparse")]
    if subset.empty:
        print(f"  Skip chart 4 ({dim}): no data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for gpu_name, grp in subset.groupby("gpu"):
        grp_sorted = grp.sort_values("rows")
        ax.plot(grp_sorted["rows"], grp_sorted["bw_efficiency_pct"],
                marker="^", label=gpu_name,
                color=COLORS.get(gpu_name, "#333"), linewidth=2)

    ax.set_xlabel("Matrix Rows")
    ax.set_ylabel("Bandwidth Efficiency (%)")
    ax.set_title(f"SpMV Bandwidth Efficiency — {dim} Poisson (cuSPARSE)")
    ax.set_xscale("log")
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Theoretical Peak")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"chart4_bw_efficiency_{dim}.png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  ✓ Chart 4 ({dim}): bw_efficiency")


# ═══════════════════════════════════════════════════════════════════════
#  Chart 5: Kernel Variant Comparison
# ═══════════════════════════════════════════════════════════════════════
def plot_kernel_comparison(bench_df, out_dir, dim="2D"):
    """Grouped bars: GB/s for each kernel, per GPU, at largest size."""
    subset = bench_df[bench_df["problem_dim"] == dim]
    if subset.empty:
        print(f"  Skip chart 5 ({dim}): no bench data")
        return

    # Pick largest problem size with block_size=256 (or 0 for cusparse)
    subset = subset[((subset["block_size"] == 256) | (subset["block_size"] == 0))]
    idx = subset.groupby(["gpu", "kernel_variant"])["grid_n"].idxmax()
    subset = subset.loc[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = subset.pivot_table(index="kernel_variant", columns="gpu",
                                values="spmv_gbps", aggfunc="first")
    pivot.plot(kind="bar", ax=ax, color=[COLORS.get(c, "#333") for c in pivot.columns])

    ax.set_ylabel("SpMV Bandwidth (GB/s)")
    ax.set_title(f"Kernel Variant Comparison — {dim} Poisson (largest size, bs=256)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="GPU")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"chart5_kernel_comparison_{dim}.png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  ✓ Chart 5 ({dim}): kernel_comparison")


# ═══════════════════════════════════════════════════════════════════════
#  Chart 6: Block Size Sweep
# ═══════════════════════════════════════════════════════════════════════
def plot_block_size_sweep(bench_df, out_dir, dim="2D"):
    """GB/s vs block size for each custom kernel, at the largest problem."""
    subset = bench_df[(bench_df["problem_dim"] == dim) &
                       (bench_df["kernel_variant"] != "cusparse")]
    if subset.empty:
        print(f"  Skip chart 6 ({dim}): no custom kernel data")
        return

    # Use largest grid_n per kernel
    largest_n = subset.groupby("kernel_variant")["grid_n"].max()
    rows = []
    for kv, max_n in largest_n.items():
        rows.append(subset[(subset["kernel_variant"] == kv) & (subset["grid_n"] == max_n)])
    subset = pd.concat(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for kv, grp in subset.groupby("kernel_variant"):
        grp_sorted = grp.sort_values("block_size")
        ax.plot(grp_sorted["block_size"], grp_sorted["spmv_gbps"],
                marker="D", label=kv,
                color=COLORS.get(kv, "#333"), linewidth=2)

    ax.set_xlabel("Block Size (threads per block)")
    ax.set_ylabel("SpMV Bandwidth (GB/s)")
    ax.set_title(f"Block Size Sweep — {dim} Poisson (largest size)")
    ax.set_xticks([64, 128, 256, 512])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"chart6_block_size_sweep_{dim}.png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  ✓ Chart 6 ({dim}): block_size_sweep")


# ═══════════════════════════════════════════════════════════════════════
#  Chart 7: 4090 vs 3080 Speedup Ratio
# ═══════════════════════════════════════════════════════════════════════
def plot_speedup(df, out_dir, dim="2D"):
    """Speedup factor (3080_time / 4090_time) vs problem size."""
    subset = df[(df["problem_dim"] == dim) & (df["kernel_variant"] == "cusparse")]
    if subset.empty:
        print(f"  Skip chart 7 ({dim}): no data")
        return

    gpus = subset["gpu"].unique()
    # Need exactly two GPUs for comparison
    gpu_4090 = [g for g in gpus if "4090" in g]
    gpu_3080 = [g for g in gpus if "3080" in g]
    if not gpu_4090 or not gpu_3080:
        print(f"  Skip chart 7 ({dim}): need both 4090 and 3080 data")
        return

    d1 = subset[subset["gpu"] == gpu_4090[0]].set_index("grid_n")["total_time_ms"]
    d2 = subset[subset["gpu"] == gpu_3080[0]].set_index("grid_n")["total_time_ms"]
    common = d1.index.intersection(d2.index).sort_values()

    if common.empty:
        print(f"  Skip chart 7 ({dim}): no common problem sizes")
        return

    speedup = d2[common] / d1[common]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(common, speedup, marker="o", color="#76B900", linewidth=2, label="Measured Speedup")
    # Theoretical bandwidth ratio line (4090: ~1008 GB/s, 3080: ~760 GB/s)
    bw_ratio = 1008.0 / 760.0
    ax.axhline(y=bw_ratio, color="gray", linestyle="--", alpha=0.7,
               label=f"BW Ratio ({bw_ratio:.2f}×)")
    ax.set_xlabel("Grid Size N")
    ax.set_ylabel("Speedup (4090 vs 3080)")
    ax.set_title(f"4090 vs 3080 Speedup — {dim} Poisson (cuSPARSE)")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"chart7_speedup_{dim}.png"), dpi=FIG_DPI)
    plt.close(fig)
    print(f"  ✓ Chart 7 ({dim}): speedup")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Generate project charts from CSV data")
    parser.add_argument("--dir", default="[results/rtx3080]")
    parser.add_argument("--out", default="[path to scripts/report/figures]")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading CG solver CSVs...")
    cg_df = load_cg_csvs(args.dir)

    print("Loading SpMV bench CSVs...")
    bench_df = load_bench_csvs(args.dir)

    print("\nGenerating charts...")

    if not cg_df.empty:
        for dim in ["2D", "3D"]:
            plot_time_vs_size(cg_df, args.out, dim)
            plot_bandwidth_vs_size(cg_df, args.out, dim)
            plot_bw_efficiency(cg_df, args.out, dim)
            plot_speedup(cg_df, args.out, dim)
        plot_time_breakdown(cg_df, args.out)

    if not bench_df.empty:
        for dim in ["2D", "3D"]:
            plot_kernel_comparison(bench_df, args.out, dim)
            plot_block_size_sweep(bench_df, args.out, dim)

    print("\nDone. Charts saved to:", args.out)


if __name__ == "__main__":
    main()
