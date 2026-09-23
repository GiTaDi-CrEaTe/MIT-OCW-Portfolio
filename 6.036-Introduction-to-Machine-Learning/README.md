# 6.036 --  Introduction to Machine Learning
*Foundations Lab: Optimization Dynamics and Gradient Precision*

---

### Core Question
How sensitive is gradient-based optimization to finite-difference precision and parameter initialization across reproducible random seeds?

### The Mathematical Guarantee
- **Multivariate Chain Rule (Backpropagation):** The gradient of the scalar loss $J$ with respect to layer weights $W_l$ is computed by backward propagation of error signals:
  $$\frac{\partial J}{\partial W_l} = \frac{\partial J}{\partial Z_l} A_{l-1}^T, \quad \frac{\partial J}{\partial Z_{l-1}} = \left(W_l^T \frac{\partial J}{\partial Z_l}\right) \odot \sigma'(Z_{l-1})$$
- **Softmax / Sigmoid Cross-Entropy Simplification:** When cross-entropy loss is paired with sigmoid or softmax logits, the output gradient simplifies algebraically to $\frac{\partial J}{\partial Z_L} = A_L - Y$.
- **Universal Approximation Theorem:** Stacking non-linear hidden layers allows the network to represent non-linear decision boundaries that single-layer linear models provably cannot separate.

### What the Code Investigates
[`neural_network_from_scratch.py`](./Applied-Theory/neural_network_from_scratch.py) implements a multi-layer feedforward network from first principles using NumPy alone (no PyTorch, no autograd):
1. **Hand-Derived Forward & Backward Passes:** Every Jacobian, error signal, and weight update is explicitly coded from calculus.
2. **Numerical Gradient Checking:** Centered finite-difference verification comparing analytical gradients against:
 $$g_{\text{num}} = \frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2\epsilon}$$
3. **Linear Separation Failure vs. Deep Resolution:** Trains a single-layer logistic regression model against a multi-layer non-linear network on concentric two-rings data.

### Empirical Findings & Failure Modes
- **Analytical Gradient Agreement:** Hand-derived backpropagation matched centered finite differences with relative error $< 10^{-7}$ in the optimal step-size regime.
- **The Finite-Difference U-Curve:** Sweeping $\epsilon \in [10^{-1}, 10^{-15}]$ revealed that $\epsilon = 10^{-15}$ causes catastrophic floating-point cancellation (100% error), while $\epsilon = 10^{-1}$ is corrupted by $O(\epsilon^2)$ truncation error. The optimal balance occurs at $\epsilon \approx 10^{-5}$ to $10^{-6}$. (See [Capstone Experiment 4](../capstone/README.md#experiment-4--finite-difference-precision-u-curve--multi-seed-ml-stability)).
- **Multi-Seed Initialization Stability:** Evaluated over 50 random seeds:
  - **Xavier Normal:** Mean accuracy **$100.0\% \pm 0.00\%$** (all 50 seeds converged).
  - **Zero Initialization:** Stuck at **$50.00\% \pm 0.00\%$** (symmetry prevents hidden units from differentiating).
  - **Oversized Initialization ($\sigma=5.0$):** High variance **$89.43\% \pm 7.85\%$** due to tanh saturation.

### Capstone & Cross-Course Connection
Backpropagation uses matrix operations from 18.06, its binary cross-entropy loss is the negative log-likelihood derived in 6.041, and its numerical stability limits are analyzed in [Capstone Experiment 4](../capstone/README.md#experiment-4--finite-difference-precision-u-curve--multi-seed-ml-stability).
