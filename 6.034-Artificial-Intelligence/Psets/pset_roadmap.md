# 6.034 — Problem Set Roadmap

## Unit 1 — Search

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 1 | State-space search framing, uninformed search | Modeling a problem as states + actions + goal test, BFS/DFS as special cases | The hardest part of this pset is never the search algorithm itself -- it's correctly defining what a "state" is for a given problem. |
| Pset 2 | Informed search — greedy best-first, A* | Heuristic functions, admissibility, the optimality proof for A* | Implemented from scratch in this folder's Applied-Theory script, along with the admissibility check that the proof depends on. |
| Pset 3 | Heuristic design | Manhattan/Euclidean distance heuristics, consistency vs. admissibility | A heuristic can be admissible without being consistent, but consistency (a stronger condition) is what guarantees A* never has to re-expand a node -- worth keeping the two conditions distinct. |

## Unit 2 — Adversarial Search

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 4 | Minimax | Game trees, backward induction over MAX/MIN layers | Minimax is backward induction from game theory wearing a computer-science hat -- same idea as solving a game by reasoning from the end state backward. |
| Pset 5 | Alpha-beta pruning | Pruning correctness proof (a pruned subtree cannot change the root's minimax value), best/worst-case pruning bounds | Implemented from scratch in Applied-Theory with an explicit node-count comparison against plain minimax on the identical game tree. |

## Unit 3 — Constraint Satisfaction

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 6 | CSP formulation, backtracking search | Variables, domains, constraints; chronological backtracking | Graph coloring (from 6.042's graph theory unit) is the canonical CSP example -- this pset makes the connection explicit. |
| Pset 7 | Constraint propagation | Forward checking, arc consistency (AC-3) | Implemented from scratch in Applied-Theory as forward checking, with an empirical comparison of nodes-visited against naive backtracking. |

## Unit 4 — Learning (overlap with 6.036, treated briefly here)

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 8 | Identification trees, nearest neighbors | Information gain, decision boundaries as partition of the state space | Conceptually reframes supervised learning (6.036's main subject) as another kind of search — searching over a space of hypotheses rather than a space of states. Not separately re-implemented in this folder's Applied-Theory script since it overlaps directly with 6.036. |

## Applied-Theory connection

`Applied-Theory/search_and_csp.py` implements Pset 2 (A*, with the admissibility condition explicitly checked and compared against Dijkstra for optimality), Pset 5 (alpha-beta pruning, benchmarked node-for-node against plain minimax on the same tree), and Pset 7 (forward-checking CSP solver for graph coloring, benchmarked against naive backtracking).
