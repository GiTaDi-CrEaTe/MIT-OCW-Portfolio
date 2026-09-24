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
