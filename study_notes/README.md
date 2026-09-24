# Study Notes

Personal problem set roadmaps for each of the six MIT OCW courses in this portfolio. Each file tracks what each pset covers, the key techniques, and my own notes on what was conceptually hardest and where the ideas connect to the code.

These aren't polished summaries. They're working notes I kept while going through the material, organized after the fact so I could trace which problem sets feed into which implementations.

## How to read these

Each roadmap is structured as a table with four columns:

| Column | What it contains |
|---|---|
| **Pset** | Problem set number from the OCW course |
| **Topic** | What the pset covers |
| **Key techniques** | The specific methods and tools used |
| **Notes** | My own observations: what was hard, what connected to other courses, what I wish I'd known earlier |

At the bottom of each file there's an **Applied-Theory connection** section that maps specific psets to the code I wrote.

## The six roadmaps

### [6.042J: Mathematics for Computer Science](./6.042-Mathematics-for-Computer-Science_pset_roadmap.md)
14 problem sets across four units: Proofs, Structures, Counting, and Discrete Probability. The number theory psets (5-6) are the direct foundation for the RSA implementation. The discrete probability unit (psets 13-14) is the on-ramp into 6.041. The single most reused idea from this course turned out to be linearity of expectation without independence assumptions (Pset 14).

### [18.06: Linear Algebra](./18.06-Linear-Algebra_pset_roadmap.md)
12 problem sets in four parts: Solving Linear Systems, Orthogonality and Projections, Eigenvalues and Eigenvectors, and the SVD. The roadmap shows how Psets 7 (Gram-Schmidt), 9 (QR eigensolver), 11 (Markov/PageRank), and 12 (SVD) chain together into one Applied-Theory script where each algorithm builds on the last. The "column picture" of $Ax = b$ (Pset 1) is the single most useful reframing in the course.

### [6.006: Introduction to Algorithms](./6.006-Introduction-to-Algorithms_pset_roadmap.md)
12 problem sets covering foundations, sorting and trees, graphs, and dynamic programming. The AVL tree (Pset 5) and graph algorithms (Psets 7-9) are implemented from scratch with empirical complexity benchmarks. The sorting lower bound proof via decision trees (Pset 3) is more subtle than any individual sorting algorithm.

### [6.041: Probabilistic Systems Analysis](./6.041-Probabilistic-Systems-Analysis_pset_roadmap.md)
11 problem sets from probability axioms through stochastic processes. The Bayesian inference pset (7) and Markov chain pset (11) are both implemented in the Applied-Theory script. The stationary distribution computation reuses the eigenvector machinery from 18.06 but now interpreted probabilistically rather than just algebraically.

### [6.036: Introduction to Machine Learning](./6.036-Introduction-to-Machine-Learning_pset_roadmap.md)
10 problem sets from linear regression through neural networks. Psets 7-8 (forward prop, backprop) are implemented in full with every gradient hand-derived and verified against finite differences. The closed-form regression solution (Pset 2) is literally the projection matrix from 18.06, re-derived by setting the gradient to zero.

### [6.034: Artificial Intelligence](./6.034-Artificial-Intelligence_pset_roadmap.md)
8 problem sets on search, adversarial games, and constraint satisfaction. A* (Pset 2), alpha-beta pruning (Pset 5), and forward-checking CSP (Pset 7) are all implemented from scratch and benchmarked against their naive counterparts. The hardest part of the search psets was never the algorithm itself; it was correctly defining what a "state" is for a given problem.

## Cross-course connections I noticed

Some patterns kept showing up across multiple courses:

- **Linearity of expectation** (6.042 Pset 14) reappears in the hashing analysis (6.006 Pset 6) and forms the backbone of probabilistic algorithm analysis throughout 6.041.
- **The projection formula** $P = A(A^TA)^{-1}A^T$ from 18.06 Psets 5-6 is exactly the least squares solution in 6.036 Pset 2.
- **Graph coloring** from 6.042 Pset 9 is the canonical example in 6.034's CSP unit (Pset 6).
- **Eigenvectors for $\lambda = 1$** solve both PageRank (18.06 Pset 11) and Markov steady states (6.041 Pset 11). Same computation, different interpretation.
- **Induction and loop invariants** from 6.042 Psets 3-4 are the proof technique for every correctness argument in 6.006 (BST invariant, AVL height bound, Dijkstra correctness).
- **MLE from 6.041** (Pset 8) is exactly the cross-entropy loss minimization in 6.036 (Pset 6). The loss function isn't an arbitrary design choice; it falls out of the Bernoulli log-likelihood.
