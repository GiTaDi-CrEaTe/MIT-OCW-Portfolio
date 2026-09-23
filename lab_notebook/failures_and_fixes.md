# Lab Notebook: Hardest Failures, Mathematical Diagnoses, and Fixes

*A documented record of algorithmic breakdowns encountered during this investigation, the mathematical root causes diagnosed, and the engineering remedies implemented.*

---

## Log Entry 1: The Illusion of Zero Residual in Classical Gram-Schmidt
- **Module:** `18.06-Linear-Algebra` & `capstone/numerical_stability.py`
- **Symptom:**
  When testing the Classical Gram-Schmidt (CGS) QR factorization on an ill-conditioned Hilbert matrix, the backward reconstruction error was near machine epsilon:
  $$\frac{\|A - Q R\|_2}{\|A\|_2} \approx 4.4 \times 10^{-17}$$
  The self-test initially checked only this residual and reported that QR decomposition had "passed."
- **Failure:**
  When checking the orthogonality of the resulting basis:
  $$\|Q^T Q - I\|_2 = 3.0016 \quad (\text{Expected } < 10^{-12})$$
  The columns of $Q$ were completely non-orthogonal! In fact, multiple columns had inner products exceeding $0.8$.
- **Mathematical Diagnosis:**
  CGS computes the projection of column $a_k$ using the formula:
  $$v_k = a_k - \sum_{j=1}^{k-1} (q_j^T a_k) q_j$$
  In floating-point arithmetic, when $a_k$ is nearly in the span of $\{q_1, \dots, q_{k-1}\}$, the vector $v_k$ is the difference of two nearly identical quantities. This triggers **catastrophic cancellation**, wiping out significant bits. The resulting vector $v_k$ is dominated by floating-point roundoff, which is then normalized to unit length, introducing an erroneous vector into the basis.
- **The Fix:**
  1. Replaced CGS with **Modified Gram-Schmidt (MGS)**:
   $$v_k^{(1)} = a_k, \quad v_k^{(j+1)} = v_k^{(j)} - \left(q_j^T v_k^{(j)}\right) q_j$$
   MGS orthogonalizes against the current working vector, removing roundoff error introduced in earlier steps.
  2. Orthogonality error dropped from **$3.00$ to $1.8 \times 10^{-4}$** on the Hilbert benchmark.
  3. Added strict tests asserting both backward residual AND basis orthogonality.

---

## Log Entry 2: SVD Obliterates Small Singular Values via $A^TA$
- **Module:** `18.06-Linear-Algebra` & `capstone/svd_investigation.py`
- **Symptom:**
  A from-scratch SVD built on the eigendecomposition of $A^TA$ worked reliably for small matrices with condition numbers $\kappa(A) \le 10^3$. When tested on a matrix with condition number $\kappa(A) = 10^9$ and true singular values $\sigma \in [1.0, 10^{-9}]$, the computed smallest singular value was:
  $$\hat{\sigma}_{\text{min}} = 0.00000000 \quad (100\% \text{ relative error})$$
  Subsequent recovery of left singular vectors $u_n = \frac{1}{\sigma_n} A v_n$ failed with `ZeroDivisionError` or returned a column of pure zeros.
- **Mathematical Diagnosis:**
  The condition number of the product $A^T A$ is the square of the condition number of $A$:
  $$\kappa(A^TA) = \left(\frac{\sigma_1}{\sigma_n}\right)^2 = \kappa(A)^2$$
  For $\kappa(A) = 10^9$, $\kappa(A^TA) = 10^{18}$.
  In IEEE 754 float64 arithmetic, the machine epsilon is $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$.
  When forming $A^TA = \sum_i \sigma_i^2 v_i v_i^T$, the smallest eigenvalue $\sigma_n^2 = 10^{-18}$ is smaller than $\epsilon_{\text{mach}} \cdot \sigma_1^2 = 2.22 \times 10^{-16}$.
  Floating-point addition rounds $\sigma_1^2 + \sigma_n^2$ to $\sigma_1^2$. The smallest eigenvalue is physically obliterated into machine noise before the eigensolver is even called!
- **The Fix:**
  1. Identified that this is an unavoidable structural limitation of any algorithm that forms the normal equations.
  2. Clamped negative eigenvalues from numerical noise (`np.maximum(eigvals, 0.0)`).
  3. Added the condition number squaring demonstration to the capstone, documenting why direct bidiagonalization (Golub-Kahan) is mandatory for $\kappa(A) \ge 10^8$.

---

## Log Entry 3: The Finite-Difference Precision Trap ($\epsilon = 10^{-15}$)
- **Module:** `6.036-Introduction-to-Machine-Learning` & `capstone/gradient_precision.py`
- **Symptom:**
  While writing the automated unit test for backpropagation gradients, I initially set the finite-difference step size to $\epsilon = 10^{-15}$, thinking: *"The definition of derivative is the limit as $\epsilon \to 0$, so smaller must be more accurate."*
  The test failed catastrophically:
  $$\text{Relative Error} = 1.0000 \quad (100\% \text{ error!})$$
  The numerical gradient evaluated to exactly `0.00000000`, while the analytical backprop gradient was `8.7302e-04`.
- **Mathematical Diagnosis:**
  The finite-difference formula is:
  $$g_{\text{num}} = \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$$
  When $\epsilon = 10^{-15}$ and $\theta \approx 1.0$:
  $$\text{fl}(\theta + \epsilon) = \text{fl}(1.0 + 10^{-15}) = 1.0$$
  In IEEE 754 float64 (53 bits of mantissa), the distance between $1.0$ and the next representable float is $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$. At $\epsilon = 10^{-15}$, $\theta + \epsilon$ and $\theta - \epsilon$ round to the exact same floating-point number.
  Therefore:
  $$f(\theta + \epsilon) - f(\theta - \epsilon) = 0.0 \implies g_{\text{num}} = 0.0$$
- **The Fix:**
  Analyzed the total error balance:
  $$\text{Total Error} \approx \underbrace{\frac{M}{6} \epsilon^2}_{\text{Truncation}} + \underbrace{\frac{2 \epsilon_{\text{mach}}}{\epsilon}}_{\text{Cancellation}}$$
  Differentiating with respect to $\epsilon$ yields the optimal step size:
  $$\epsilon^* \approx \left(\frac{6 \epsilon_{\text{mach}}}{M}\right)^{1/3} \approx 10^{-5} \text{ to } 10^{-6}$$
  At $\epsilon = 10^{-5}$, the relative error dropped to **$2.79 \times 10^{-9}$**. Mapped this into [Capstone Experiment 4](capstone/README.md#experiment-4--finite-difference-precision-u-curve--multi-seed-ml-stability).

---

