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

The capstone experiments (documented in [`capstone/`](./capstone/README.md), figures in [`artifacts/`](./artifacts/)) test four questions across the six courses:

| Question | What the textbook says | What actually happens | Figure |
|---|---|---|---|
| **Is from-scratch linear algebra stable?** | Gram-Schmidt produces $Q^TQ = I$ | Classical GS loses orthogonality ($\|Q^TQ - I\|_2 > 1.0$) at $\kappa(A) \ge 10^9$; Modified GS holds up | [Fig 1](./artifacts/fig1_gram_schmidt_orthogonality.png) |
| | SVD via $A^TA$: $\sigma_i = \sqrt{\lambda_i(A^TA)}$ | Squaring the condition number zeroes out small singular values when $\kappa(A) \ge 10^8$ | [Fig 2](./artifacts/fig2_svd_condition_squaring.png) |
| **Can a heuristic cut search without breaking optimality?** | Admissible $h(n) \le h^*(n)$ guarantees shortest paths | Overestimating by 1.5x cuts nodes by 90% but fails 63.7% of the time; tie-breaking cuts 96% with 0% failure | [Fig 3](./artifacts/fig3_astar_search_efficiency.png) |
| **How precise is gradient computation?** | $\lim_{\epsilon \to 0} \frac{\Delta f}{2\epsilon} = \nabla f$ | U-curve: too-small $\epsilon$ causes catastrophic cancellation, optimal only near $10^{-5}$ | [Fig 4](./artifacts/fig4_gradient_finite_difference_u_curve.png) |
| **What happens to Bayesian updating when the model is wrong?** | Posterior concentrates on the true parameter | A static model tracking a switching coin reports 95% certainty while being wrong 95.8% of the time | [Fig 5](./artifacts/fig5_model_misspecification.png) |
| **Do the six subjects connect?** | Each course stands alone | Integer rings ($\mathbb{Z}/n\mathbb{Z}$) have zero error; float64 can't even do $(10^{16}+1)-10^{16}-1$ correctly | [Synthesis](./capstone/cross_course_synthesis.py) |

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
| |-- run_experiments.py     # Runs all experiments and regenerates artifacts
| |-- numerical_stability.py   # Gram-Schmidt breakdown study
| |-- svd_investigation.py   # Condition number squaring trap
| |-- search_efficiency.py   # A* heuristic scaling study
| |-- gradient_precision.py    # Finite-difference U-curve
| |-- model_misspecification.py  # Bayesian overconfidence under drift
| |-- cross_course_synthesis.py  # How the six courses connect
| +-- README.md
|
|-- artifacts/         # Generated figures from capstone experiments
|-- tests/         # pytest suite covering all implementations
|-- _study_notes/        # Problem set roadmaps and personal study notes
|-- lab_notebook/        # Log of hardest failures and how I fixed them
| +-- failures_and_fixes.md
|-- LIMITATIONS.md       # Honest accounting of what this code can't do
|-- requirements.txt
+-- .github/workflows/ci.yml    # CI across Python 3.11, 3.12, 3.13
```

---

## Study Notes

The `_study_notes/` directory contains problem set roadmaps for each course. These are my personal notes on what each pset covers, what techniques matter, and where the ideas connect to the implementations. They're organized by course:

- [**18.06 Linear Algebra**](./_study_notes/18.06-Linear-Algebra_pset_roadmap.md): 12 psets from solving $Ax = b$ through eigenvalues to the SVD. The roadmap traces how Psets 7, 9, 11, and 12 chain together into the Applied-Theory script.
- [**6.042 Discrete Math**](./_study_notes/6.042-Mathematics-for-Computer-Science_pset_roadmap.md): 14 psets covering proofs, number theory, counting, and discrete probability. Psets 5-6 (Euclidean algorithm, Fermat's Little Theorem) feed directly into the RSA implementation.
- [**6.006 Algorithms**](./_study_notes/6.006-Introduction-to-Algorithms_pset_roadmap.md): 12 psets from asymptotic notation through sorting, BSTs, graphs, and dynamic programming. The AVL tree (Pset 5) and Dijkstra (Pset 9) are implemented from scratch.
- [**6.041 Probability**](./_study_notes/6.041-Probabilistic-Systems-Analysis_pset_roadmap.md): 11 psets from axioms through random variables, Bayesian inference, and Markov chains. The Bayesian updater (Pset 7), MLE comparison (Pset 8), and Markov simulation (Pset 11) are all in the Applied-Theory script.
- [**6.036 Machine Learning**](./_study_notes/6.036-Introduction-to-Machine-Learning_pset_roadmap.md): 10 psets from linear regression and gradient descent through neural networks. Psets 7-8 (forward prop, backprop) are implemented in full with numerical gradient verification.
- [**6.034 Artificial Intelligence**](./_study_notes/6.034-Artificial-Intelligence_pset_roadmap.md): 8 psets covering search, adversarial games, and constraint satisfaction. A* (Pset 2), alpha-beta (Pset 5), and forward-checking CSP (Pset 7) are implemented and benchmarked.

Each roadmap includes a section at the bottom explaining exactly which psets map to which parts of the code.

---

## Running the Code

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full test suite
pytest -v

# Run the capstone experiments and regenerate all figures
python3 capstone/run_experiments.py
```

CI runs automatically via [GitHub Actions](.github/workflows/ci.yml) on Python 3.11, 3.12, and 3.13.

---

## Failures, Limitations, and Honest Scope

I'm listing these up front because I think knowing what went wrong matters as much as showing what works.

- **[Lab Notebook: Failures and Fixes](./lab_notebook/failures_and_fixes.md):** The six hardest bugs I ran into during this project, with full mathematical diagnoses. Includes: the illusion of zero residual in CGS, condition number squaring in SVD, catastrophic cancellation in finite differences, Python recursion limits on degenerate BSTs, the A* open-space plateau, and Bayesian overconfidence under regime switching.

- **[LIMITATIONS.md](./LIMITATIONS.md):** An honest list of what this code does not do and why. Textbook RSA vs. OAEP padding, Gram-Schmidt vs. Householder reflectors, unshifted QR vs. Francis QR, conjugate priors vs. MCMC, pure NumPy vs. GPU tensor libraries, grid A* vs. continuous motion planning. Each gap is explained with the mathematical reason, not just "this is left as future work."

---

## What I Learned

The biggest lesson wasn't about any single algorithm. It was that mathematical correctness and computational correctness are different things, and the gap between them is where most real engineering problems live. A proof says Gram-Schmidt works. It does work, in exact arithmetic. But IEEE 754 isn't exact arithmetic, and understanding why it fails (catastrophic cancellation in the projection step) is what lets you pick Modified GS or Householder reflections instead.

The second thing that stuck with me: small backward error does not imply a correct answer. My CGS code produced $\|A - QR\| \approx 10^{-17}$ while the Q matrix was completely non-orthogonal. If I had only checked the residual, I would have shipped broken code with confidence.

---

## Open Questions

Some things I ran into that I still don't have clean answers for:

- **Why does MGS work better than CGS when they're mathematically identical?** I can explain the mechanism (sequential vs. original projections, catastrophic cancellation), but I don't have a tight error bound of the form $\|Q^TQ - I\| \le f(\kappa, \epsilon_{\text{mach}})$ for MGS that I've derived myself. The literature says $O(\kappa \cdot \epsilon_{\text{mach}})$ for MGS vs. $O(\kappa^2 \cdot \epsilon_{\text{mach}})$ for CGS, but I haven't worked through the proof.
- **Is there a principled way to pick the discount factor $\gamma$ in the adaptive Bayesian model?** I tuned it by hand. There should be a way to learn the switching rate from the data itself (maybe a hidden Markov model), but that's a much harder inference problem.
- **The tie-breaking trick for A* on open grids feels like a hack.** It works, and it preserves admissibility, but I wonder if there's a deeper geometric reason why $f$-cost plateaus form on uniform grids, and whether there's a systematic way to break them that generalizes beyond Manhattan distance.

---

## What's Next

There are parts of these courses I haven't implemented yet, and there are natural extensions:

- **Dynamic programming** (6.006 Psets 11-12): knapsack, edit distance, shortest paths as DP
- **Householder QR** (18.06): the algorithm that production solvers actually use, and why it doesn't care about condition numbers
- **MCMC sampling** (6.041): extending beyond conjugate priors to handle real posterior distributions
- **Convolutional layers** (6.036): adding structure to the from-scratch network
- **MCTS** (6.034): the search algorithm that actually works for large game trees (Go, Chess)
- **Shifted QR with Wilkinson shifts** (18.06): why the unshifted QR eigensolver is slow on clustered eigenvalues

I'm also interested in combining the Markov chain work from 6.041 with the neural network from 6.036 to build a simple reinforcement learning agent, which would connect all six courses into one system.

---

## Resources Used

- **Lectures and problem sets:** MIT OpenCourseWare (ocw.mit.edu) for all six courses
- **Primary textbooks:** Strang's *Introduction to Linear Algebra* (18.06), CLRS *Introduction to Algorithms* (6.006), Bertsekas & Tsitsiklis *Introduction to Probability* (6.041)
- **Numerical analysis reference:** Trefethen & Bau, *Numerical Linear Algebra*, for understanding why CGS fails and Householder doesn't
- **Implementation:** Python 3.11+, NumPy (for array operations only, not for `numpy.linalg` solvers), Matplotlib (for figures), pytest (for testing)

