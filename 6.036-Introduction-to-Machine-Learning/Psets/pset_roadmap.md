# 6.036 — Problem Set Roadmap

## Unit 1 — Linear Methods

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 1 | The learning problem, hypothesis classes, loss functions | Empirical risk minimization framing | Reframing "fit a model" as "minimize an explicit loss function over a hypothesis class" is the conceptual move the entire course is built on. |
| Pset 2 | Linear regression | Closed-form least squares (normal equations), connection to 18.06's projection formula | The closed-form solution is literally the projection matrix from 18.06 Pset 5-6, re-derived here by setting the gradient of squared error to zero. |
| Pset 3 | Gradient descent | Batch GD, stochastic GD, learning rate, convergence conditions for convex losses | The convexity argument for why GD converges to the *global* minimum on linear regression's squared-error loss (not just a local one) is the pset's real content. |
| Pset 4 | Regularization | Ridge regression (L2), the bias-variance tradeoff | Ridge regression's closed form differs from OLS by exactly one extra term (λI) added before inverting — a small algebraic change with a large effect on generalization. |

## Unit 2 — Classification

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 5 | The perceptron | Mistake-driven learning, the perceptron convergence theorem | The convergence theorem (bounded number of mistakes if data is linearly separable) is a genuinely elegant proof — geometric margin arguments doing real work. |
| Pset 6 | Logistic regression | Sigmoid link function, cross-entropy loss, MLE derivation of the loss | Direct continuation of 6.041's MLE material — the cross-entropy loss isn't an arbitrary design choice, it falls out of maximizing the Bernoulli log-likelihood. |

## Unit 3 — Neural Networks

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 7 | Feedforward networks, forward propagation | Layer composition, activation functions (sigmoid, tanh, ReLU) | The universal approximation intuition (why stacking non-linear layers can represent decision boundaries a single linear layer cannot) motivated the non-linear classification demo in Applied-Theory. |
| Pset 8 | Backpropagation | Chain rule applied layer-by-layer, computing gradients w.r.t. every weight and bias | Implemented fully from scratch in Applied-Theory, with a numerical gradient check to verify every hand-derived partial derivative. |
| Pset 9 | Training dynamics | Mini-batching, weight initialization, vanishing/exploding gradients | The initialization-scale argument (why weights can't be initialized as all-zero, and why naive large-scale initialization causes saturation) came up directly while debugging the from-scratch network. |

## Unit 4 — Unsupervised Learning (brief)

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 10 | Clustering | k-means, the alternating-minimization view (assign step / update step each provably decrease the objective) | Not implemented in Applied-Theory in this portfolio, but conceptually connects to the SVD-based dimensionality ideas in 18.06 as another "find structure without labels" tool. |

## Applied-Theory connection

`Applied-Theory/neural_network_from_scratch.py` implements Pset 7-8 in full: forward propagation through an arbitrary number of layers, backpropagation with every gradient hand-derived via the chain rule, and — critically — a finite-difference gradient check (the standard debugging technique taught in Pset 8 for verifying a from-scratch backprop implementation) confirming the analytical gradients are correct before they're ever used for training.
