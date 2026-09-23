# 6.034 --  Artificial Intelligence
*Foundations Lab: Heuristic Search and Constraint Bounds*

---

### Core Question
How much can admissible heuristic information reduce search space without sacrificing path optimality, and what happens when the heuristic overestimates?

### The Mathematical Guarantee
- **A\* Optimality Theorem:** If heuristic $h(n)$ is admissible ($h(n) \le h^*(n)$ for all $n$) and consistent ($h(u) \le c(u, v) + h(v)$), A\* search is guaranteed to return an optimal shortest path without expanding any node whose $f$-cost exceeds the optimal path length $C^*$.
- **Alpha-Beta Equivalence:** Alpha-beta pruning returns the exact same minimax value as exhaustive minimax while pruning subtrees that provably cannot affect the root decision, cutting effective branching from $b$ down to $\approx \sqrt{b}$ in the best case.
- **Arc Consistency (AC-3):** Enforcing arc consistency prunes domain values that have no valid support, detecting unsolvability before backtracking begins.

### What the Code Investigates
[`search_and_csp.py`](./Applied-Theory/search_and_csp.py) implements classical search and constraint reasoning from scratch:
1. **A\* vs. Dijkstra on Obstacle Fields:** Measures node expansions and path lengths across Manhattan and Euclidean heuristics.
2. **Minimax & Alpha-Beta Game Trees:** Benchmarks node evaluations on identical game trees to confirm pruning correctness.
3. **Constraint Satisfaction with Forward Checking:** Solves map coloring problems and benchmarks node visits against naive backtracking.

### Empirical Findings & Failure Modes
- **Heuristic Search Reduction:** Across 80 trials on 50×50 grids, Manhattan distance with tie-breaking reduced search space from **2500 nodes (Dijkstra) down to 99-287 nodes (85%-96% reduction)** while achieving **0% suboptimality**.
- **Inadmissibility Breakdown:** When testing an overestimating heuristic ($1.5 \times h_M$), search was rapid but returned suboptimal paths on **63.7% of all runs**, demonstrating how fragile the optimality guarantee is. (See [Capstone Experiment 3](../capstone/README.md#experiment-3--a-heuristic-search-scaling--optimality-limits)).
- **Alpha-Beta Node Reduction:** Alpha-beta pruning achieved identical root game values to plain minimax while visiting **41.7% fewer nodes** on random game trees.
- **Forward Checking Pruning:** Forward checking solved 18-vertex 3-coloring problems exploring **68.2% fewer search states** than naive chronological backtracking.

### Capstone & Cross-Course Connection
A* search reframes the graph representations and priority queues of 6.006 with heuristic guidance. The full scaling study across obstacle densities and grid dimensions is presented in [Capstone Experiment 3](../capstone/README.md#experiment-3--a-heuristic-search-scaling--optimality-limits).
