# Foundations Lab: From Proof to Experiment

> **Disclaimer:** *Independent study based on publicly available MIT OpenCourseWare materials. Not affiliated with or endorsed by MIT.*

I taught myself six MIT courses through OCW and built every algorithm from scratch, not to reimplement what already exists in NumPy or SciPy, but to find out where the textbook math breaks on real hardware. Textbooks prove that Gram-Schmidt orthogonalizes vectors. They don't mention that on a computer, it quietly stops working when the matrix is ill-conditioned, and the residual still looks fine.

This repo is the result of roughly eight months of independent study, about 4,000 lines of from-scratch Python, 67 problem sets worth of notes, six capstone experiments, and more debugging sessions than I'd like to admit.

---

## Why I Built This

I kept hitting a wall in my coursework. I could follow a proof on paper, solve the problem set, and get the right answer. But I didn't really understand what was happening underneath. When I tried to implement Gram-Schmidt from scratch, it produced garbage orthogonality on a Hilbert matrix while reporting a tiny residual. That single failure taught me more about numerical analysis than any lecture.

So I decided to do this systematically: take six foundational MIT courses, implement the core algorithms from the ground up (no black-box library calls for the things I'm studying), and then deliberately stress-test them until they break. Every failure is documented in the [lab notebook](./lab_notebook/failures_and_fixes.md). Every known limitation is cataloged in [LIMITATIONS.md](./LIMITATIONS.md).

The central question:

$$\text{Mathematical Guarantee} \quad \xrightarrow{\quad\text{IEEE 754 Float64 \& Finite Sampling}\quad} \quad \text{What actually happens on a computer?}$$

---

## What I Found

The capstone experiments (documented in [`capstone/`](./capstone/README.md), figures in [`artifacts/`](./artifacts/)) test four core questions and two generalization challenges across the six courses:

| Question | What the textbook says | What actually happens | Figure |
|---|---|---|---|
| **Is from-scratch linear algebra stable?** | Gram-Schmidt produces $Q^TQ = I$ | Classical GS loses orthogonality ($\|Q^TQ - I\|_2 > 0.42$) at $\kappa(A) \ge 10^8$; Modified GS holds up ($10^{-9}$) | [Fig 1](./artifacts/fig1_gram_schmidt_orthogonality.png) |
| | From-scratch SVD via $A^TA$ vs LAPACK baseline | Squaring the condition number zeroes out small singular values when $\kappa(A) \ge 10^8$ (354% error at $10^9$) | [Fig 2](./artifacts/fig2_svd_condition_squaring.png) |
| **Can a heuristic cut search without breaking optimality?** | Admissible $h(n) \le h^*(n)$ guarantees shortest paths | Overestimating by 1.5x cuts nodes but fails 63.7% of the time; lexicographic tie-breaking cuts 96% with 0% failure | [Fig 3](./artifacts/fig3_astar_search_efficiency.png) |
| **How precise is gradient computation?** | $\lim_{\epsilon \to 0} \frac{\Delta f}{2\epsilon} = \nabla f$ | U-curve: too-small $\epsilon$ causes catastrophic cancellation, optimal only near $10^{-5}$ | [Fig 4](./artifacts/fig4_gradient_finite_difference_u_curve.png) |
| **What happens to Bayesian updating when the model is wrong?** | Posterior concentrates on the true parameter | A static model tracking a switching coin reports 95% certainty while being wrong 95.8% of the time | [Fig 5](./artifacts/fig5_model_misspecification.png) |
| **Can a unified metric predict computational breakdown?** | Each course stands alone | The Computational Reliability Index (CRI) $\rho \in [0, 1]$ unifies error margins across all six domains | [Fig 6](./artifacts/fig6_computational_reliability.png) |
| **Does the reliability index generalize to unseen algorithms?** | Metrics tuned on training data overfit | Holdout validation on Dataset B (Cholesky, Householder, 4th-order FD, complex step, mazes, unpivoted LU) yields AUROC 0.949, F1 0.897 | [Fig 7](./artifacts/fig7_cri_holdout_validation.png) |
| **Does it predict breakdown in external production libraries?** | Production solvers are trusted black boxes | On $H_{13}$, `scipy.linalg.solve` yields 1580% error with $10^{-16}$ residual; CRI detects breakdown while residuals fail | [Fig 8](./artifacts/fig8_cri_external_validation.png) |

---

## The Bigger Picture: When Theory Meets Hardware

Foundations Lab is not an isolated exercise in working through problem sets. The central question here -- what actually happens when mathematical theorems hit real computational constraints -- is the same thread running through almost everything else I build:

- **Synth Guard (AI Media Detection):** Machine learning models are trained assuming clean, stationary data. In the real world, adversarial noise and compression break those assumptions immediately. I built Synth Guard to handle model misspecification in synthetic media detection instead of trusting brittle decision boundaries.
- **aero-kv (High-Performance Storage):** An algorithm textbook promises $O(1)$ hash table lookups or $O(\log n)$ tree searches. But real hardware has CPU cache hierarchies, memory fences, and tail latency spikes. In aero-kv, I worked through what it actually takes to build storage engines with predictable latency and crash-consistency invariants.
- **AI & Systems Experiments:** Exploring scheduling, memory protection, and deterministic resource isolation when agent runtimes run on constrained local machines.

Instead of building disconnected projects, I keep coming back to the same theme: finding where systems break down under real-world constraints, and figuring out how to engineer around the failure.

---


## The Six Courses

| Course | What I investigated | Key implementation | What surprised me |
|---|---|---|---|
| [**6.042J: Discrete Math**](./6.042-Mathematics-for-Computer-Science) | Why modular arithmetic gives exact guarantees that floating-point can't | [RSA from scratch](./6.042-Mathematics-for-Computer-Science/Applied-Theory/number_theory_cryptography.py): Euclid, Bezout, Miller-Rabin, 512-bit RSA | Zero precision loss ($\|m_{\text{recovered}} - m\| = 0$), but Python's variable-time BigInt arithmetic leaks timing info. |
| [**18.06: Linear Algebra**](./18.06-Linear-Algebra) | When linear algebra becomes numerically unstable | [Matrix decompositions from scratch](./18.06-Linear-Algebra/Applied-Theory/linear_algebra_from_scratch.py): LU, CGS/MGS QR, QR eigensolver, SVD, PageRank | CGS orthogonality error hits $3.0$ on Hilbert matrices while the residual stays at $10^{-17}$. A small residual does NOT mean a correct answer. |
| [**6.006: Algorithms**](./6.006-Introduction-to-Algorithms) | How structural invariants protect data structures | [AVL tree + graph algorithms from scratch](./6.006-Introduction-to-Algorithms/Applied-Theory/graph_algorithms_and_data_structures.py): AVL, BFS, DFS, Dijkstra | AVL holds height at $14 \le 1.45\log_2(5000)$ on sorted input; the same keys in an unbalanced BST degenerate to a linked list of height 5000. |
| [**6.041: Probability**](./6.041-Probabilistic-Systems-Analysis) | What happens to Bayesian updating when model assumptions are wrong | [Bayesian inference + Markov chains from scratch](./6.041-Probabilistic-Systems-Analysis/Applied-Theory/bayesian_inference_and_markov_chains.py): Beta-Binomial updating, MLE, Markov simulation | 500K-step Markov simulation matches the spectral stationary distribution ($\|\Delta\| < 0.003$); static models produce false certainty on drifting data. |
| [**6.036: Machine Learning**](./6.036-Introduction-to-Machine-Learning) | How sensitive gradient optimization is to step size and initialization | [Neural network from scratch](./6.036-Introduction-to-Machine-Learning/Applied-Theory/neural_network_from_scratch.py): NumPy MLP, hand-derived backprop, gradient checks | Hand-derived gradients match finite differences to $< 10^{-7}$; Xavier init gets $100.0\% \pm 0.0\%$ across 50 seeds, zero init gets stuck at 50%. |
| [**6.034: Artificial Intelligence**](./6.034-Artificial-Intelligence) | How much search can be eliminated by heuristic information | [Search and CSP from scratch](./6.034-Artificial-Intelligence/Applied-Theory/search_and_csp.py): A* with multiple heuristics, Alpha-Beta minimax, AC-3 CSP | Manhattan + tie-breaking reduces search by up to 96% with 0% suboptimality; overestimating heuristics fail in 63.7% of runs. |

---

## How the Courses Connect

This isn't six isolated projects. Each course builds on the ones before it:

```
6.042 (Discrete Proofs & Invariants)   --> Exact integer rings, induction, invariant maintenance
 |
 v
18.06 (Continuous Linear Algebra)    --> Vector spaces, decompositions, condition number limits
 |
 v
6.006 (Algorithms & Data Structures)   --> Rebalancing invariants (AVL), graph traversal
 |
 v
6.041 (Probability & Inference)    --> Stochastic processes, Markov eigen-dynamics, Bayesian limits
 |
 v
6.036 (Machine Learning)      --> 18.06 Jacobians + 6.041 MLE loss -> hand-derived backprop
 |
 v
6.034 (Artificial Intelligence)    --> 6.006 graphs + 6.042 invariants -> heuristic A* & CSPs
```

---

## Repository Map

```
MIT-OCW-Portfolio/
|
|-- 6.042-Mathematics-for-Computer-Science/
| |-- Applied-Theory/number_theory_cryptography.py
| +-- README.md
|
|-- 18.06-Linear-Algebra/
| |-- Applied-Theory/linear_algebra_from_scratch.py
| +-- README.md
|
|-- 6.006-Introduction-to-Algorithms/
| |-- Applied-Theory/graph_algorithms_and_data_structures.py
| +-- README.md
|
|-- 6.034-Artificial-Intelligence/
| |-- Applied-Theory/search_and_csp.py
| +-- README.md
|
|-- 6.036-Introduction-to-Machine-Learning/
| |-- Applied-Theory/neural_network_from_scratch.py
| +-- README.md
|
|-- 6.041-Probabilistic-Systems-Analysis/
| |-- Applied-Theory/bayesian_inference_and_markov_chains.py
| +-- README.md
|
|-- capstone/        # Cross-course experiments with reproducible figures
| |-- run_experiments.py     # Runs all 8 experiments and regenerates artifacts
| |-- numerical_stability.py   # Gram-Schmidt breakdown study
| |-- svd_investigation.py   # Condition number squaring trap
| |-- search_efficiency.py   # A* heuristic scaling study
| |-- gradient_precision.py    # Finite-difference U-curve
| |-- model_misspecification.py  # Bayesian overconfidence under drift
| |-- cross_course_synthesis.py  # How the six courses connect
| |-- cri.py                 # Mathematical formulation of multi-axis CRI
| |-- cri_validation.py      # Holdout validation (Dataset A vs Dataset B)
| |-- cri_external.py        # External library challenge (SciPy/LAPACK)
