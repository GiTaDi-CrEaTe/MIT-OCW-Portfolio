"""
Foundations Lab  --  Capstone Experiment 4: Floating-Point Gradient Precision & Training Stability
==============================================================================================

Research Questions:
  1. How sensitive is numerical gradient verification to the finite-difference step size epsilon?
  2. How sensitive is neural network training to weight initialization across 50 random seeds?

Mathematical Mechanics:
  Part 1  --  The Finite-Difference U-Curve (Truncation vs Cancellation):
    Centered difference approximation of df/d theta:
      g_num(theta; eps) = [f(theta + eps) - f(theta - eps)] / (2 * eps)
    Taylor expansion reveals two competing error regimes:
      - Truncation error:
          [f'''(theta) / 6] * eps^2 = O(eps^2)   (dominates when eps is large, e.g. 10^-1)
      - Floating-point cancellation error:
          Because f(theta + eps) ~= f(theta - eps), computing their difference loses significant bits:
          |fl(f(theta + eps)) - fl(f(theta - eps))| / (2 * eps) = O(eps_mach / eps)
          (dominates when eps is tiny, e.g. 10^-14, where division by eps blows up roundoff error).
    The total relative error forms a classical U-curve with optimal step size:
      eps* ~= (eps_mach)^(1/3) ~= (2.22e-16)^(1/3) ~= 6e-6 ~ 1e-5 to 1e-7.

  Part 2  --  Initialization and Convergence Across 50 Random Seeds:
    We evaluate a 2-hidden layer network on the non-linear two-rings benchmark:
      - He/Xavier Normal Initialization: maintains activation variance across layers.
      - Zero Initialization: symmetry problem  --  all hidden units receive identical gradients.
      - Oversized Initialization (sigma = 5.0): causes severe tanh saturation (activations near +/- 1),
        collapsing gradients to near-zero (tanh'(z) = 1 - tanh(z)^2 -> 0).
    Report mean +/- std accuracy over 50 independent trials.
"""

from typing import Dict, List, Tuple
import numpy as np


class MiniMLP:
    """Configurable Multi-Layer Perceptron using float64 NumPy."""
    def __init__(self, layer_dims: List[int], init_type: str = "xavier", seed: int = 42):
        self.L = len(layer_dims) - 1
        self.params: Dict[str, np.ndarray] = {}
        rng = np.random.default_rng(seed)

        for l in range(1, self.L + 1):
            n_in, n_out = layer_dims[l - 1], layer_dims[l]
            if init_type == "xavier":
                scale = np.sqrt(2.0 / (n_in + n_out))
                self.params[f"W{l}"] = rng.standard_normal((n_out, n_in), dtype=np.float64) * scale
            elif init_type == "zero":
                self.params[f"W{l}"] = np.zeros((n_out, n_in), dtype=np.float64)
            elif init_type == "oversized":
                self.params[f"W{l}"] = rng.standard_normal((n_out, n_in), dtype=np.float64) * 5.0
            else:
                raise ValueError(f"Unknown init_type: {init_type}")
            self.params[f"b{l}"] = np.zeros((n_out, 1), dtype=np.float64)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        cache: Dict[str, np.ndarray] = {"A0": X}
        A = X
        for l in range(1, self.L + 1):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = W @ A + b
            cache[f"Z{l}"] = Z
            if l < self.L:
                A = np.tanh(Z)
            else:
                # Sigmoid for binary classification
                A = 1.0 / (1.0 + np.exp(-np.clip(Z, -30.0, 30.0)))
            cache[f"A{l}"] = A
        return A, cache

    def compute_loss(self, A_L: np.ndarray, Y: np.ndarray) -> float:
        eps = 1e-15
        A_clipped = np.clip(A_L, eps, 1.0 - eps)
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(A_clipped) + (1.0 - Y) * np.log(1.0 - A_clipped)) / m
        return float(loss)

    def backward(self, Y: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        grads: Dict[str, np.ndarray] = {}
        m = Y.shape[1]
        A_L = cache[f"A{self.L}"]
        dZ = A_L - Y  # Exact gradient of BCE with sigmoid output

        for l in range(self.L, 0, -1):
            A_prev = cache[f"A{l-1}"]
            grads[f"dW{l}"] = (dZ @ A_prev.T) / m
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m

            if l > 1:
                dA_prev = self.params[f"W{l}"].T @ dZ
                Z_prev = cache[f"Z{l-1}"]
                # d/dz tanh(z) = 1 - tanh(z)^2
                dZ = dA_prev * (1.0 - np.tanh(Z_prev) ** 2)

        return grads

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 1000, lr: float = 0.5) -> List[float]:
        losses = []
        for _ in range(epochs):
            A_L, cache = self.forward(X)
            loss = self.compute_loss(A_L, Y)
            losses.append(loss)
            grads = self.backward(Y, cache)
            for l in range(1, self.L + 1):
                self.params[f"W{l}"] -= lr * grads[f"dW{l}"]
                self.params[f"b{l}"] -= lr * grads[f"db{l}"]
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        A_L, _ = self.forward(X)
        return (A_L >= 0.5).astype(int)


def generate_two_rings_data(n_per_ring: int = 150, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates concentric two-rings dataset."""
    rng = np.random.default_rng(seed)
    theta_in = rng.uniform(0, 2 * np.pi, n_per_ring)
    r_in = rng.normal(1.0, 0.12, n_per_ring)
    inner = np.stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)])

    theta_out = rng.uniform(0, 2 * np.pi, n_per_ring)
    r_out = rng.normal(2.5, 0.15, n_per_ring)
    outer = np.stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)])

    X = np.concatenate([inner, outer], axis=1)
    Y = np.concatenate([np.zeros((1, n_per_ring)), np.ones((1, n_per_ring))], axis=1)
    return X, Y


def run_finite_difference_precision_sweep() -> Dict[str, List]:
    """
    Sweeps epsilon from 10^-1 to 10^-15 to demonstrate the U-shaped error curve
    balancing truncation error and floating-point cancellation.
    """
    X, Y = generate_two_rings_data(n_per_ring=20, seed=42)
    net = MiniMLP([2, 4, 1], init_type="xavier", seed=42)

    # Compute analytical gradients via backprop
    A_L, cache = net.forward(X)
    analytic_grads = net.backward(Y, cache)

    # We evaluate a specific weight element W1[0, 0]
    analytic_val = analytic_grads["dW1"][0, 0]

    epsilons = [10**(-p) for p in range(1, 16)]
    relative_errors = []
    numerical_grads = []

    for eps in epsilons:
        orig = net.params["W1"][0, 0]

        # f(theta + eps)
        net.params["W1"][0, 0] = orig + eps
        A_plus, _ = net.forward(X)
        L_plus = net.compute_loss(A_plus, Y)

        # f(theta - eps)
        net.params["W1"][0, 0] = orig - eps
        A_minus, _ = net.forward(X)
        L_minus = net.compute_loss(A_minus, Y)

        # Restore
        net.params["W1"][0, 0] = orig

        num_grad = (L_plus - L_minus) / (2.0 * eps)
        rel_err = abs(num_grad - analytic_val) / (abs(num_grad) + abs(analytic_val) + 1e-15)

        numerical_grads.append(num_grad)
        relative_errors.append(rel_err)

    return {
        "epsilons": epsilons,
        "analytic_val": analytic_val,
        "numerical_grads": numerical_grads,
        "relative_errors": relative_errors,
    }


def run_50_seed_training_experiment(num_seeds: int = 50) -> Dict[str, Dict[str, float]]:
    """
    Evaluates training stability across 50 independent seeds under 3 initialization schemes:
      1. Xavier Normal (recommended)
      2. Zero Initialization (symmetry failure)
      3. Oversized Normal (sigma = 5.0, gradient saturation)
    """
    X, Y = generate_two_rings_data(n_per_ring=150, seed=123)

    inits = ["xavier", "zero", "oversized"]
    summary: Dict[str, Dict[str, float]] = {}

    for init in inits:
        accuracies = []
        final_losses = []
        for s in range(num_seeds):
            net = MiniMLP([2, 12, 8, 1], init_type=init, seed=s)
            losses = net.train(X, Y, epochs=600, lr=0.3)
            preds = net.predict(X)
            acc = float(np.mean(preds == Y))
            accuracies.append(acc)
            final_losses.append(losses[-1])

        summary[init] = {
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "max_accuracy": float(np.max(accuracies)),
            "mean_loss": float(np.mean(final_losses)),
            "std_loss": float(np.std(final_losses)),
        }

    return summary


if __name__ == "__main__":
    print("=" * 80)
    print("EXPERIMENT 4A: Finite-Difference Epsilon Precision Sweep (U-Curve)")
    print("=" * 80)
    sweep = run_finite_difference_precision_sweep()
    print(f"Analytical Gradient Reference: {sweep['analytic_val']:.8e}")
    print(f"{'Epsilon':>10} | {'Numerical Grad':>18} | {'Relative Error':>16} | {'Regime'}")
    print("-" * 80)
    for eps, g_num, err in zip(sweep["epsilons"], sweep["numerical_grads"], sweep["relative_errors"]):
        if eps >= 1e-3:
            regime = "Truncation Error Dominated O(eps^2)"
        elif eps <= 1e-11:
            regime = "Cancellation Error Dominated O(eps_mach/eps)"
        else:
            regime = "Optimal Precision Balance"
        print(f"{eps:10.1e} | {g_num:18.8e} | {err:16.4e} | {regime}")

    print("\n" + "=" * 80)
    print("EXPERIMENT 4B: Training Reproducibility Across 50 Random Seeds")
    print("=" * 80)
    seeds_res = run_50_seed_training_experiment(num_seeds=50)
    print(f"{'Initialization':<15} | {'Mean Accuracy':>14} | {'Std Dev':>10} | {'Min Acc':>9} | {'Max Acc':>9}")
    print("-" * 80)
    for init_name, stats in seeds_res.items():
        print(
            f"{init_name:<15} | {stats['mean_accuracy']*100:13.2f}% | "
            f"{stats['std_accuracy']*100:9.2f}% | {stats['min_accuracy']*100:8.2f}% | {stats['max_accuracy']*100:8.2f}%"
        )

