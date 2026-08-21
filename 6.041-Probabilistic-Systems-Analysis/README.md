# 6.041 — Probabilistic Systems Analysis and Applied Probability

**MIT OCW subject:** 6.041/6.431, EECS.

## What this course is

6.041 builds probability up rigorously from the axioms: sample spaces and events, discrete and continuous random variables, joint/conditional distributions, Bayes' rule, the law of total probability, expectation and variance, the weak law of large numbers, the central limit theorem, and an introduction to Markov chains. Where 6.042 treats probability as a counting exercise, 6.041 treats it as the mathematics of *inference under uncertainty* — the course's real subject is how to update beliefs correctly given evidence.

## Why it matters for this portfolio

This is the course that makes 6.036's probabilistic framing of machine learning precise:
- Maximum likelihood estimation (used to justify the loss functions in 6.036) is a direct application of this course's estimation theory.
- Bayesian updating here is the same machinery behind Bayesian methods in ML, done without the "machine learning" framing so the underlying logic is visible.
- Markov chains here are the same object as the transition matrix used for PageRank in the 18.06 folder — this course supplies the probabilistic interpretation, 18.06 supplies the linear-algebra computation.

## What I focused on

The `Applied-Theory/` script implements three things end to end: (1) Bayesian inference from scratch on a discrete hypothesis space, showing posterior concentration as evidence accumulates; (2) maximum likelihood estimation for a Bernoulli parameter, compared against the Bayesian posterior mean to make the "MLE is the large-sample limit of the Bayesian estimate" relationship concrete rather than asserted; and (3) a discrete-time Markov chain simulator whose empirical long-run state frequencies are checked against the analytically computed stationary distribution.

## Folder contents

- [`Psets/pset_roadmap.md`](./Psets/pset_roadmap.md) — topic-by-topic syllabus breakdown.
- [`Applied-Theory/bayesian_inference_and_markov_chains.py`](./Applied-Theory/bayesian_inference_and_markov_chains.py) — discrete Bayesian updating, MLE vs. posterior-mean comparison, and Markov chain simulation vs. stationary-distribution ground truth.
