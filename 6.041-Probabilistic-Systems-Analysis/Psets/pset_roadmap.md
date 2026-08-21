# 6.041 — Problem Set Roadmap

## Unit 1 — Probability Foundations

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 1 | Sample spaces, probability axioms | Kolmogorov axioms, counting-based probability | Continuation of 6.042's discrete probability unit, now formalized with the three axioms as the actual foundation rather than intuition. |
| Pset 2 | Conditional probability, Bayes' rule | Law of total probability, Bayes' theorem derivation | The derivation of Bayes' rule from the *definition* of conditional probability (not just stating it) is what makes it reusable outside toy problems. |
| Pset 3 | Independence | Pairwise vs. mutual independence, common pitfalls | The gap between pairwise and full independence is a genuinely easy-to-miss trap — worth an entire pset on its own. |

## Unit 2 — Random Variables

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 4 | Discrete random variables | PMFs, expectation, variance | Builds directly on 6.042's counting unit; expectation here is where "average value" gets a rigorous definition via the PMF. |
| Pset 5 | Continuous random variables | PDFs, CDFs, common distributions (uniform, exponential, normal) | The PDF-as-density (not probability) distinction is the single most common conceptual error at this stage — worth being pedantic about. |
| Pset 6 | Multiple random variables | Joint/marginal/conditional distributions, covariance | Sets up the multivariate machinery needed for any real statistical estimation problem. |

## Unit 3 — Inference and Limit Theorems

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 7 | Bayesian inference | Prior, likelihood, posterior; conjugate priors (Beta-Bernoulli) | Implemented from scratch in this folder's Applied-Theory script on a discrete grid of hypotheses. |
| Pset 8 | Classical (frequentist) inference | Maximum likelihood estimation, bias, consistency | Directly compared against the Bayesian posterior mean in Applied-Theory to show how MLE emerges as a limiting case. |
| Pset 9 | Weak law of large numbers, central limit theorem | Convergence in probability, Chebyshev's inequality as the proof mechanism | The WLLN proof via Chebyshev is a clean, short, and genuinely convincing proof — a good example of a "small" inequality doing a lot of work. |

## Unit 4 — Stochastic Processes

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 10 | Bernoulli and Poisson processes | Memorylessness, interarrival times | The memoryless property of the exponential distribution is the bridge from discrete Bernoulli trials to continuous-time arrival processes. |
| Pset 11 | Markov chains | Transition matrices, stationary distributions, classification of states (recurrent/transient) | Implemented and empirically verified in Applied-Theory; directly reuses the eigenvector-based stationary-distribution computation from the 18.06 folder, now interpreted probabilistically rather than just algebraically. |

## Applied-Theory connection

`Applied-Theory/bayesian_inference_and_markov_chains.py` implements Pset 7 (Bayesian updating), Pset 8 (MLE, directly compared against the Bayesian result), and Pset 11 (Markov chains, verified by long-run simulation against the analytically solved stationary distribution — the same underlying computation as the PageRank script in `18.06-Linear-Algebra/Applied-Theory`, here derived and validated from the probability side rather than the algebra side).
