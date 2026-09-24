# 6.041 --  Probabilistic Systems Analysis
*Foundations Lab: Bayesian Inference, Estimation, and Stochastic Limits*

---

### Core Question
How does Bayesian inference behave when the data-generating process violates the structural assumptions of the model?

### The Mathematical Guarantee
- **Conjugate Posterior Updating:** A Beta prior $\text{Beta}(\alpha, \beta)$ updated with $k$ successes out of $n$ Bernoulli trials yields an exact conjugate posterior $\text{Beta}(\alpha + k, \beta + n - k)$.
- **Bernstein-von Mises Theorem:** As $n \to \infty$, the Bayesian posterior distribution concentrates around the Maximum Likelihood Estimate (MLE) $\hat{\theta}_{\text{MLE}} = \frac{k}{n}$, washing out the influence of any continuous prior.
- **Ergodic Theorem for Markov Chains:** For an irreducible, aperiodic discrete-time Markov chain with transition matrix $P$, the long-run empirical state frequency $\frac{1}{T} \sum_{t=1}^T \mathbb{I}(X_t = s)$ converges almost surely to the unique stationary distribution $\pi$ satisfying $\pi P = \pi$.

### What the Code Investigates
[`bayesian_inference_and_markov_chains.py`](./Applied-Theory/bayesian_inference_and_markov_chains.py) implements core probability mechanisms from scratch:
1. **Discrete & Conjugate Bayesian Updating:** Tracks posterior evolution and variance concentration under sequential Bernoulli evidence.
2. **MLE vs. Bayesian Posterior Mean Comparison:** Demonstrates how the Bayesian estimate approaches the frequentist sample mean as sample size grows.
3. **Markov Chain Simulation vs. Spectral Solution:** Solves the linear system $\pi (P - I) = 0$ with $\sum \pi_i = 1$ and checks empirical convergence over 500,000 simulated transitions.

### Empirical Findings & Failure Modes
- **Convergence to Stationary Equilibrium:** A 500,000-step simulation of a 3-state weather Markov chain matched the exact spectral stationary distribution with maximum absolute error $< 0.003$.
- **Bernstein-von Mises Agreement:** At $n = 2000$ trials, the MLE estimate ($0.7025$) agreed with the Bayesian posterior mean to within $0.005$.
- **Model Misspecification & False Certainty:** When observing non-stationary coin flips driven by a regime-switching Markov chain, the static i.i.d. Bayesian model accumulated false certainty, narrowing its credible interval around an unrepresentative average and yielding an empirical coverage of only **4.2%**! (Investigated thoroughly in [Capstone Experiment 5](../capstone/README.md#experiment-5--probability-under-model-misspecification)).

### Capstone & Cross-Course Connection
The stationary distribution $\pi P = \pi$ is an eigenvector problem that directly reuses 18.06 eigensolvers. At the same time, Bernoulli negative log-likelihood minimization here is identical to the binary cross-entropy loss function used to train 6.036 neural networks.
