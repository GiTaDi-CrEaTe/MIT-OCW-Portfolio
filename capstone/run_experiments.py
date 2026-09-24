"""
Foundations Lab  --  Capstone Experiment Runner
==============================================

Runs all eight capstone experiments, verifies failure boundaries,
and generates figures in `artifacts/`.

Usage:
  python3 capstone/run_experiments.py
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from capstone.numerical_stability import run_condition_number_sweep, run_hilbert_experiment
from capstone.svd_investigation import run_svd_condition_experiment
from capstone.search_efficiency import run_benchmark_sweep
from capstone.gradient_precision import (
    run_finite_difference_precision_sweep,
    run_50_seed_training_experiment,
)
from capstone.model_misspecification import (
    generate_regime_switching_stream,
    run_bayesian_iid_updating,
    run_bayesian_adaptive_updating,
    run_nonstationary_experiment,
    run_heavy_tail_experiment,
)
from capstone.cross_course_synthesis import (
    verify_discrete_vs_continuous_precision,
    synthesize_eigensolver_and_markov_chain,
    synthesize_loss_and_gradient_cancellation,
    evaluate_computational_reliability_index,
)
from capstone.cri_validation import run_holdout_validation
from capstone.cri_external import run_external_challenge



def ensure_artifacts_dir() -> Path:
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def plot_fig1_gram_schmidt(sweep: dict, artifacts_dir: Path):
    plt.figure(figsize=(8, 5))
    conds = sweep["cond_numbers"]
    plt.loglog(conds, sweep["cgs_ortho"], "r-o", label="Classical Gram-Schmidt (CGS)", linewidth=2, markersize=6)
    plt.loglog(conds, sweep["mgs_ortho"], "b-s", label="Modified Gram-Schmidt (MGS)", linewidth=2, markersize=6)
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Complete Loss of Orthogonality (||Q^TQ - I|| = 1)")
    plt.xlabel(r"Condition Number $\kappa(A)$", fontsize=11)
    plt.ylabel(r"Orthogonality Error $\|Q^T Q - I\|_2$", fontsize=11)
    plt.title("Figure 1: Catastrophic Loss of Orthogonality in Gram-Schmidt", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    out_path = artifacts_dir / "fig1_gram_schmidt_orthogonality.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Artifact] Saved: {out_path}")


def plot_fig2_svd(svd_res: dict, artifacts_dir: Path):
    plt.figure(figsize=(8, 5))
    conds = svd_res["cond_numbers"]
    plt.loglog(conds, svd_res["ata_sv_rel_error"], "r-^", label=r"From-Scratch $A^TA$ SVD", linewidth=2, markersize=6)
    plt.loglog(conds, svd_res["direct_sv_rel_error"], "g-o", label=r"LAPACK Direct SVD Baseline (dgesdd)", linewidth=2, markersize=6)
    plt.axvline(1e8, color="purple", linestyle="--", alpha=0.7, label=r"Critical Threshold $\kappa(A) \approx 10^8 \Rightarrow \kappa(A^TA) \approx 1/\epsilon_{mach}$")
    plt.xlabel(r"Condition Number $\kappa(A)$", fontsize=11)
    plt.ylabel(r"Smallest Singular Value Relative Error $|\hat{\sigma}_n - \sigma_n| / \sigma_n$", fontsize=11)
    plt.title("Figure 2: Condition Number Squaring and Precision Loss in SVD", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    out_path = artifacts_dir / "fig2_svd_condition_squaring.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Artifact] Saved: {out_path}")


def plot_fig3_search(search_data: dict, artifacts_dir: Path):
    records = [r for r in search_data["records"] if r["size"] == 50]
    densities = [r["density"] for r in records]
    dijkstra = [r["dijkstra_nodes"] for r in records]
    euclidean = [r["euclidean_nodes"] for r in records]
    manhattan = [r["manhattan_nodes"] for r in records]
    tiebreak = [r["tiebreak_nodes"] for r in records]

    plt.figure(figsize=(8.5, 5))
    bar_width = 0.02
    d_arr = np.array(densities)
    plt.bar(d_arr - 1.5 * bar_width, dijkstra, width=bar_width, label="Dijkstra (h=0)", color="#7f7f7f")
    plt.bar(d_arr - 0.5 * bar_width, euclidean, width=bar_width, label="A* (Euclidean)", color="#1f77b4")
    plt.bar(d_arr + 0.5 * bar_width, manhattan, width=bar_width, label="A* (Manhattan)", color="#2ca02c")
    plt.bar(d_arr + 1.5 * bar_width, tiebreak, width=bar_width, label="A* (Manhattan + Lexicographic Tie-Break)", color="#d62728")

    plt.xlabel("Obstacle Density", fontsize=11)
    plt.ylabel("Mean Nodes Expanded (50x50 Grid)", fontsize=11)
    plt.title("Figure 3: Search Space Reduction across Obstacle Fields", fontsize=12, fontweight="bold")
    plt.xticks(densities, [f"{d:.2f}" for d in densities])
    plt.grid(True, axis="y", ls=":", alpha=0.6)
    plt.legend(fontsize=9)
    plt.tight_layout()
    out_path = artifacts_dir / "fig3_astar_search_efficiency.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Artifact] Saved: {out_path}")


def plot_fig4_gradient(grad_res: dict, artifacts_dir: Path):
    plt.figure(figsize=(8, 5))
    eps = grad_res["epsilons"]
    err = grad_res["relative_errors"]
    plt.loglog(eps, err, "k-o", linewidth=2, markersize=5)
    plt.axvline(1e-5, color="green", linestyle=":", alpha=0.8, label=r"Optimal Step Size $\epsilon^* \approx \epsilon_{mach}^{1/3} \approx 6\times 10^{-6}$")
    plt.annotate(
        r"Truncation Error $O(\epsilon^2)$",
        xy=(1e-2, 1e-4),
        xytext=(1e-3, 1e-2),
        arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
        fontsize=10,
        color="blue",
    )
    plt.annotate(
        r"Cancellation Error $O(\epsilon_{mach}/\epsilon)$",
        xy=(1e-13, 1e-1),
        xytext=(1e-15, 1e-3),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        fontsize=10,
        color="red",
    )
    plt.xlabel(r"Finite-Difference Step Size $\epsilon$", fontsize=11)
    plt.ylabel(r"Relative Error $\|g_{num} - g_{analytic}\| / \|g\|$", fontsize=11)
    plt.title("Figure 4: The Finite-Difference Precision U-Curve", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    out_path = artifacts_dir / "fig4_gradient_finite_difference_u_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
