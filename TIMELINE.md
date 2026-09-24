# Research Development Timeline

*By Adityajyoti Kar (GitHub: GiTaDi-CrEaTe)*

This timeline documents the chronological development of the Foundations Lab project, spanning roughly eight months of independent study.

## Phase 1: Foundation and Problem Discovery (Months 1-2)
- Started with 6.042 discrete math (number theory, proofs, invariants)
- Implemented RSA from scratch, discovered the stark difference between exact integer arithmetic and floating-point representations
- Moved to 18.06 linear algebra, implemented Gram-Schmidt orthogonalization
- FIRST MAJOR FAILURE: Classical Gram-Schmidt (CGS) produced garbage orthogonality on the Hilbert matrix while falsely reporting a tiny residual
- This failure became the central research question of the project

## Phase 2: Core Implementations and Data Structures (Months 3-4)
- 6.006 algorithms: AVL trees, graph algorithms, structural invariants
- Discovered Python recursion limits on degenerate binary search trees
- 6.041 probability: Bayesian inference, Markov chains
- Found that static Bayesian models produce false certainty on drifting data

## Phase 3: Cross-Domain Investigation (Months 5-6)
- 6.036 machine learning: Neural network from scratch, backpropagation, gradient checking
- Discovered the finite-difference U-curve (setting epsilon too small gives 100% error due to catastrophic cancellation)
- 6.034 artificial intelligence: A* search, minimax, CSP
- Discovered the f-cost plateau problem and solved it with lexicographic tie-breaking

## Phase 4: Synthesis and the CRI Hypothesis (Month 7)
- Designed the six capstone experiments
- Ran controlled condition-number sweeps across 14 orders of magnitude
- Noticed a pattern: every domain has a transition boundary between reliable and unreliable computation
- Formulated the Computational Reliability Index hypothesis

## Phase 5: Validation and Falsification Attempt (Month 8)
- Built holdout validation for CRI
- Tested CRI against an external system (scipy) that I did not build
