# 6.006 — Problem Set Roadmap

## Unit 1 — Foundations

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 1 | Asymptotic notation, algorithmic complexity | Big-O/Θ/Ω definitions, recurrence relations | The formal ε-N-style definition of O(f(n)) is more precise than the "drop constants" shortcut most people learn first — worth doing properly once. |
| Pset 2 | Peak-finding, divide and conquer | Recursion trees, the Master Theorem | The 1-D and 2-D peak-finding problems are the cleanest possible introduction to "divide and conquer beats brute force," before the algorithms get more complicated. |

## Unit 2 — Sorting and Trees

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 3 | Comparison sorting, sorting lower bounds | Merge sort correctness, the Ω(n log n) decision-tree argument | The lower bound proof (any comparison sort needs Ω(n log n) comparisons, via counting leaves in a decision tree) is more subtle than any individual sorting algorithm. |
| Pset 4 | Binary search trees | BST invariant, in-order traversal correctness | Sets up the height-balance problem that AVL trees solve — an unbalanced BST degrades to O(n) per operation on adversarial input. |
| Pset 5 | Balanced BSTs (AVL trees) | Rotations, height invariant maintenance, amortized-style height bound proof | Implemented from scratch in this folder's Applied-Theory script, along with an empirical height comparison against an unbalanced BST. |
| Pset 6 | Hashing | Chaining vs. open addressing, universal hashing, expected O(1) analysis via linearity of expectation | The expected-time analysis is a direct reuse of the linearity-of-expectation trick from 6.042 Pset 14. |

## Unit 3 — Graphs

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 7 | Graph representations, BFS | Adjacency list vs. matrix trade-offs, shortest paths in unweighted graphs | BFS's correctness proof (it visits nodes in non-decreasing order of distance) is a clean induction on distance layers. |
| Pset 8 | DFS, topological sort, strongly connected components | Edge classification (tree/back/forward/cross edges), Tarjan/Kosaraju | The edge-classification framework is what makes cycle detection and topological sort fall out of DFS almost for free. |
| Pset 9 | Weighted shortest paths — Dijkstra | Greedy correctness proof via cut property, priority-queue implementation | Implemented from scratch in Applied-Theory; the correctness proof (once a node is finalized, its distance is correct) is the pset's real content, not the code. |
| Pset 10 | Bellman-Ford, negative edge weights | Relaxation-based DP, handling negative cycles | Extends Dijkstra's relaxation idea to a setting where the greedy argument breaks down and dynamic programming is needed instead. |

## Unit 4 — Dynamic Programming

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 11 | DP I | Optimal substructure, memoization vs. tabulation, longest common subsequence | The "what is the subproblem" question is harder than any individual recurrence — this pset is where DP starts to feel like a design discipline rather than a trick. |
| Pset 12 | DP II | Knapsack variants, edit distance, DP on graphs (shortest paths as DP) | Connects back to Bellman-Ford: shortest-path computation is itself a dynamic program over "number of edges used." |

## Applied-Theory connection

`Applied-Theory/graph_algorithms_and_data_structures.py` implements Pset 5 (AVL trees, with an empirical O(log n) height check) and Psets 7–9 (BFS, DFS, Dijkstra on an adjacency-list graph), closing with a runtime-scaling benchmark that checks the theoretical complexity bounds against measured wall-clock time as input size grows.
