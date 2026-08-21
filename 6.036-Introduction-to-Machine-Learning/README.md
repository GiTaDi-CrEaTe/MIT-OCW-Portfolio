# 6.036 — Introduction to Machine Learning

**MIT OCW subject:** 6.036, EECS.

## What this course is

6.036 builds supervised learning up from optimization theory: linear and logistic regression as loss-minimization problems, gradient descent (batch and stochastic) with a convergence analysis, regularization, the perceptron, feedforward neural networks and backpropagation derived as repeated application of the chain rule, and a brief introduction to unsupervised learning via k-means clustering. The course is explicit that "machine learning" is mostly applied optimization plus applied linear algebra plus applied probability — which is exactly why it sits last in this repository's sequence.

## Why it matters for this portfolio

This is the course where the rest of the portfolio's tools get pointed at a single unifying task: fitting a model to data.
- Linear regression's closed-form solution is the least-squares projection from 18.06, re-derived here via calculus instead of geometry, and both derivations are compared directly in the Applied-Theory script.
- Logistic regression's loss function is a direct application of maximum likelihood estimation from 6.041.
- Backpropagation is nothing but the multivariate chain rule, applied mechanically layer by layer — the "derive it, don't just call `.backward()`" version of what `PyTorch` automates.

## What I focused on

The `Applied-Theory/` script implements a fully connected feedforward neural network **with backpropagation derived and coded by hand** — every gradient (weight, bias, and each layer's error signal) is computed from an explicit chain-rule derivation, not autograd. To prove the hand-derived gradients are actually correct (not just "the loss went down"), the script includes a **numerical gradient check**: it perturbs each parameter by a small epsilon, measures the resulting change in loss, and confirms the finite-difference approximation matches the analytically computed backprop gradient to high precision. The network is then trained on a non-linearly-separable 2D classification task (which a single-layer linear model provably cannot solve) to demonstrate why depth and non-linearity matter, not just assert it.

## Folder contents

- [`Psets/pset_roadmap.md`](./Psets/pset_roadmap.md) — topic-by-topic syllabus breakdown.
- [`Applied-Theory/neural_network_from_scratch.py`](./Applied-Theory/neural_network_from_scratch.py) — a NumPy-only MLP with hand-derived forward/backward passes, a gradient-check against finite differences, and a non-linear classification demo.
