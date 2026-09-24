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
    print(f"  [Artifact] Saved: {out_path}")


def plot_fig5_misspecification(artifacts_dir: Path):
    obs, true_thetas = generate_regime_switching_stream(n_steps=400, seed=42)
    m_means, m_stds, m_low, m_high = run_bayesian_iid_updating(obs)
    a_means, a_stds, a_low, a_high = run_bayesian_adaptive_updating(obs, discount_factor=0.95)

    steps = np.arange(len(obs))
    plt.figure(figsize=(10, 5))
    plt.plot(steps, true_thetas, "k--", label=r"True Instantaneous Parameter $\theta_t$", linewidth=2)
    plt.plot(steps, m_means, "r-", label="Misspecified i.i.d. Posterior Mean", linewidth=1.5)
    plt.fill_between(steps, m_low, m_high, color="red", alpha=0.15, label="Misspecified 95% Credible Interval")
    plt.plot(steps, a_means, "b-", label="Adaptive (Regime-Aware) Mean", linewidth=1.5)
    plt.fill_between(steps, a_low, a_high, color="blue", alpha=0.15, label="Adaptive 95% Credible Interval")

    plt.xlabel("Time Step $t$", fontsize=11)
    plt.ylabel(r"Parameter Value $\theta$", fontsize=11)
    plt.title("Figure 5: Model Misspecification  --  False Certainty in Static Bayesian Models", fontsize=12, fontweight="bold")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    out_path = artifacts_dir / "fig5_model_misspecification.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Artifact] Saved: {out_path}")


def plot_fig6_computational_reliability(cri_data: dict, artifacts_dir: Path):
    domains = list(cri_data.keys())
    safe_vals = [cri_data[d]["safe_cri"] for d in domains]
    fail_vals = [cri_data[d]["fail_cri"] for d in domains]

    short_labels = [
        "QR\nOrthogonality",
        "SVD\nConditioning",
        "Gradient\nVerification",
        "Bayesian\nInference",
        "A* Heuristic\nSearch",
        "Algebraic\nExactness",
    ]

    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(domains))
    width = 0.35

    plt.bar(x - width/2, safe_vals, width, label="Controlled / Robust Formulation (Safe)", color="#2ca02c", edgecolor="black", alpha=0.85)
    plt.bar(x + width/2, fail_vals, width, label="Naive / Uncalibrated Implementation (Failure)", color="#d62728", edgecolor="black", alpha=0.85)

    plt.axhline(0.5, color="purple", linestyle="--", linewidth=1.5, label=r"Transition Threshold ($\rho = 0.5$)")
    plt.ylabel(r"Computational Reliability Index $\rho \in [0, 1]$", fontsize=11)
    plt.title("Figure 6: The Computational Reliability Index (CRI) Spectrum across Six Domains", fontsize=12, fontweight="bold")
    plt.xticks(x, short_labels, fontsize=10)
    plt.ylim(0.0, 1.15)
    plt.grid(True, axis="y", ls=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_path = artifacts_dir / "fig6_computational_reliability.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Artifact] Saved: {out_path}")


def plot_fig7_cri_holdout(val_res: dict, artifacts_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    y_true = val_res["y_true"]
    y_scores = val_res["y_scores"]

    thresholds = np.linspace(0.0, 1.0, 101)
    tprs = []
    fprs = []
    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fp = np.sum((y_true == 0) & (pred == 1))
        fn = np.sum((y_true == 1) & (pred == 0))
        tn = np.sum((y_true == 0) & (pred == 0))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    m = val_res["metrics"]
    ci = val_res["bootstrap_ci"]
    ax1.plot(fprs, tprs, "b-", lw=2, label=f"CRI ROC (AUROC = {m['auroc']:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.500)")
    ax1.plot(m["fpr"], m["recall"], "r*", markersize=12, label=r"Operating Point ($\rho=0.5$)")
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax1.set_title("(a) ROC Curve on Held-Out Dataset B", fontsize=12, fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9)

    # Calibration diagram on right
    bins = np.linspace(0.0, 1.0, 6)
    emp_probs = []
    pred_probs = []
    for i in range(len(bins) - 1):
        mask = (y_scores >= bins[i]) & (y_scores < bins[i + 1])
        if np.sum(mask) > 0:
            pred_probs.append(float(np.mean(y_scores[mask])))
            emp_probs.append(float(np.mean(y_true[mask])))

    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    if pred_probs:
        ax2.plot(
            pred_probs,
            emp_probs,
            "s-",
            color="purple",
            lw=2,
            markersize=8,
            label=f"CRI Calibration (Brier = {m['brier_score']:.3f})",
        )
    ax2.set_xlabel("Predicted Failure Probability (1 - rho)", fontsize=11)
    ax2.set_ylabel("Empirical Failure Frequency", fontsize=11)
    ax2.set_title("(b) Probability Calibration on Unseen Domains", fontsize=12, fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    out_path = artifacts_dir / "fig7_cri_holdout_validation.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Artifact] Saved: {out_path}")


def plot_fig8_cri_external(ext_res: dict, artifacts_dir: Path):
    records = ext_res["solve_challenge"]["records"]
    conds = [r["cond"] for r in records]
    fwd_errs = [r["forward_error"] for r in records]
    resids = [r["residual_norm"] for r in records]
    bounds = [r["forward_error_bound"] for r in records]
    naive_cris = [r["naive_cri"] for r in records]
    revised_cris = [r["revised_cri"] for r in records]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Top: Forward Error vs Backward Residual
    ax1.loglog(conds, fwd_errs, "r-o", lw=2, markersize=6, label=r"Actual Forward Error $\|\hat{x} - x^*\| / \|x^*\|$")
    ax1.loglog(conds, resids, "b-s", lw=2, markersize=6, label=r"Backward Residual $\|b - A\hat{x}\| / \|b\|$")
    ax1.loglog(conds, bounds, "k--", alpha=0.6, label=r"Perturbation Bound $\kappa(A) \epsilon_{mach}$")
    ax1.axhline(1e-2, color="gray", linestyle=":", label="Failure Tolerance (1e-2)")
    ax1.set_ylabel("Error / Residual Magnitude", fontsize=11)
    ax1.set_title("Figure 8: External Library Challenge (SciPy/LAPACK Linear Solver Breakdown)", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=9)

    # Bottom: Naive vs Revised CRI
    ax2.semilogx(conds, naive_cris, "m--^", lw=2, markersize=6, label=f"Naive Residual-Only CRI (Accuracy: {ext_res['naive_accuracy']*100:.1f}%)")
    ax2.semilogx(conds, revised_cris, "g-o", lw=2, markersize=6, label=f"Revised Stability-Aware CRI (Accuracy: {ext_res['revised_accuracy']*100:.1f}%)")
    ax2.axhline(0.5, color="purple", linestyle="--", alpha=0.7, label=r"Transition Threshold ($\rho = 0.5$)")
    ax2.set_xlabel(r"Condition Number $\kappa(A)$", fontsize=11)
    ax2.set_ylabel(r"Reliability Index $\rho \in [0, 1]$", fontsize=11)
    ax2.set_ylim(-0.05, 1.15)
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.legend(loc="center left", fontsize=9)

    plt.tight_layout()
    out_path = artifacts_dir / "fig8_cri_external_validation.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Artifact] Saved: {out_path}")


def main():
    print("=" * 80)
    print("      FOUNDATIONS LAB: WHEN GUARANTEES MEET REAL MACHINES")
    print("               Complete Experimental Verification Suite")
    print("=" * 80)
    print("METADATA & ENVIRONMENT:")
    print(f"  Python Version:     {sys.version.split()[0]}")
    print(f"  NumPy Version:      {np.__version__}")
    print(f"  Machine Epsilon:    {np.finfo(np.float64).eps:.2e}")
    print("  RNG Master Seeds:   1806, 6042, 6006, 6041, 6036, 6034")
    print("=" * 80)

    artifacts_dir = ensure_artifacts_dir()

    print("\n>>> Running Experiment 1: Gram-Schmidt Numerical Stability...")
    sweep1 = run_condition_number_sweep()
    h_res = run_hilbert_experiment()
    plot_fig1_gram_schmidt(sweep1, artifacts_dir)

    print("\n>>> Running Experiment 2: From-Scratch SVD vs LAPACK Baseline...")
    svd_res = run_svd_condition_experiment()
    plot_fig2_svd(svd_res, artifacts_dir)

    print("\n>>> Running Experiment 3: A* Search Efficiency across Density Fields...")
    search_data = run_benchmark_sweep(grid_sizes=[30, 50], densities=[0.0, 0.1, 0.2, 0.25], trials_per_config=10)
    plot_fig3_search(search_data, artifacts_dir)

    print("\n>>> Running Experiment 4: Floating-Point Gradient Precision...")
    grad_res = run_finite_difference_precision_sweep()
    plot_fig4_gradient(grad_res, artifacts_dir)

    print("\n>>> Running Experiment 5: Model Misspecification & Bayesian Updating...")
    plot_fig5_misspecification(artifacts_dir)

    print("\n>>> Running Experiment 6: Cross-Course Synthesis & Reliability Index...")
    disc_res = verify_discrete_vs_continuous_precision()
    markov_res = synthesize_eigensolver_and_markov_chain()
    grad_cancel = synthesize_loss_and_gradient_cancellation()
    cri_data = evaluate_computational_reliability_index()
    plot_fig6_computational_reliability(cri_data, artifacts_dir)

    print("\n>>> Running Experiment 7: CRI Holdout Validation (Dataset A vs Dataset B)...")
    val_res = run_holdout_validation()
    plot_fig7_cri_holdout(val_res, artifacts_dir)

    print("\n>>> Running Experiment 8: External Library Challenge (SciPy/LAPACK)...")
    ext_res = run_external_challenge()
    plot_fig8_cri_external(ext_res, artifacts_dir)

    print("\n" + "=" * 80)
    print("                     ALL EXPERIMENTS COMPLETED")
    print(f"  Visual artifacts generated in: {artifacts_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

