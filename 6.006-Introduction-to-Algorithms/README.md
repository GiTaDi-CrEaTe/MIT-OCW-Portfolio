# 6.006 --  Introduction to Algorithms
*Foundations Lab: Invariant Preservation and Graph Complexity*

---

### Core Question
How do structural invariants protect algorithms from worst-case degeneration, and how closely do asymptotic wall-clock runtimes match Big-O theory?

### The Mathematical Guarantee
- **AVL Height Balance Invariant:** For every node $u$, $|h(u.\text{left}) - h(u.\text{right})| \le 1$. Structural induction proves that an AVL tree of $n$ keys has height bounded by $h \le 1.44 \log_2(n) + c$, guaranteeing $O(\log n)$ search, insert, and delete.
- **BFS Shortest Paths:** BFS visits vertices in non-decreasing order of edge-distance, guaranteeing optimal unweighted path lengths in $O(V + E)$ time.
- **Dijkstra's Greedy Invariant:** With non-negative edge weights, the vertex popped from the min-heap has already achieved its true minimal distance.

### What the Code Investigates
[`graph_algorithms_and_data_structures.py`](./Applied-Theory/graph_algorithms_and_data_structures.py) implements core data structures and graph algorithms from scratch:
1. **Self-Balancing AVL Binary Search Tree:** Implements left, right, left-right, and right-left rotations on recursive insertions.
2. **Unbalanced BST Comparison Baseline:** Evaluates pathological degeneration on adversarial sorted inputs.
3. **Graph Algorithms:** Adjacency-list representation with BFS, iterative DFS (with discovery/finish timestamps), and priority-queue Dijkstra.
4. **Empirical Asymptotic Scaling:** Benchmarks wall-clock execution time of BFS across doubling vertex counts $V \in [500, 4000]$.

### Empirical Findings & Failure Modes
- **AVL Invariant Preservation:** On $N = 5000$ sorted keys, the AVL tree maintained height $h = 14 \le 1.45 \log_2(5000)$, while the naive unbalanced BST degraded to a linear chain of height $5000$.
- **Call-Stack Overflow on Degenerate Trees:** Traversal of the un-rebalanced baseline crashed Python's call stack with `RecursionError` at depth 1000, demonstrating that invariant failure creates fatal operating system runtime failures. (See [Lab Notebook](../lab_notebook/failures_and_fixes.md#log-entry-4-python-recursion-limit-on-degenerate-bst)).
- **Asymptotic Linear Scaling:** Empirical wall-clock times for sparse graph BFS ($E \approx 4V$) scaled strictly linearly with $V$, consistent with theoretical $O(V + E)$ bounds.

### Capstone & Cross-Course Connection
The graph representations and min-heap priority queue implemented here provide the foundational data structures for 6.034's A* heuristic search and CSP constraint networks, and are evaluated under obstacle fields in [Capstone Experiment 3](../capstone/README.md#experiment-3--a-heuristic-search-scaling--optimality-limits).
