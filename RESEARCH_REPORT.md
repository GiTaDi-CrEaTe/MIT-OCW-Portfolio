# The Computational Reliability of Mathematical Guarantees: When Theory Meets Finite-Precision Machines

**Author:** Adityajyoti Kar (GitHub: GiTaDi-CrEaTe)  
**Affiliation:** Independent study based on MIT OpenCourseWare (6.042, 18.06, 6.006, 6.041, 6.036, 6.034). Not affiliated with or endorsed by MIT.  
**Execution Environment:** Python 3.11 -- 3.13, NumPy, SciPy, IEEE-754 Float64 ($\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$)  
**Artifacts & Verification:** 8 Figures in `artifacts/`, 55 Passing Pytest Unit Tests  

---

## 1. Research Question

Can computational breakdown across different mathematical disciplines be characterized and predicted by a domain-independent reliability index?

Mathematical theorems prove properties under assumptions of exact real arithmetic ($\mathbb{R}$), infinite precision, or strict model adherence. When translated into software, these guarantees encounter finite-precision hardware, heuristic approximations, and non-stationary data streams. Does computational failure happen as disconnected domain-specific accidents, or does it follow a predictable transition boundary that can be quantified by a common index?

---

## 2. Hypothesis

Computational failure across numerical, combinatorial, and probabilistic domains can be modeled through four coupled state variables:
1. Numerical Error ($e_{\text{num}}$): Relative discrepancy from exact algebraic or analytical values.
2. Decision Error ($e_{\text{dec}}$): Suboptimality penalty in discrete or combinatorial branching.
3. Assumption Violation ($v_{\text{assump}}$): Degree of divergence from theoretical premises.
4. Stability Risk ($r_{\text{stab}}$): Proximity to numerical or structural breakdown ($1 - m_{\text{stab}}$).

Under a weakest-link formulation, failure occurs when the worst component exceeds its critical tolerance $\tau_k$:
$$z = \max_k \left( \frac{x_k}{\tau_k} \right), \quad \rho = \frac{1}{1 + z^2}$$
where $\rho \in [0, 1]$ is the Computational Reliability Index (CRI).

I chose the rational quadratic curve $\rho = \frac{1}{1 + z^2}$ over the standard logistic sigmoid $\frac{1}{1 + e^z}$ for three mathematical reasons:
1. Exact upper bound: At zero normalized penalty ($z = 0$), $\rho(0) = 1.0$ exactly, representing unbroken theoretical guarantees without requiring an arbitrary offset.
2. Natural transition boundary: When the dominant risk hits its critical tolerance ($z = 1$), $\rho(1) = 0.5$ exactly, creating an unambiguous threshold between acceptable operation and failure.
3. Power-law decay: As $z \to \infty$, $\rho$ decays quadratically as $O(z^{-2})$, matching classical second-order error propagation rather than exponential vanishing, which prematurely underflows float64 values to zero.

I hypothesized that tolerances calibrated on one set of experimental domains (Dataset A) would generalize to predict failure ($\rho < 0.5$) on unseen computational algorithms and external production libraries (Dataset B) with an AUROC exceeding 0.85.

---

## 3. Methods

