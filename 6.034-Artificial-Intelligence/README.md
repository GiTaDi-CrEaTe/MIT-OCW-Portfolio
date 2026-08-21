# 6.034 — Artificial Intelligence

**MIT OCW subject:** 6.034, EECS.

## What this course is

6.034 covers the "classical" (search- and logic-based, pre-deep-learning) core of AI: uninformed and informed search (BFS/DFS, greedy best-first, A*), adversarial search for two-player games (minimax, alpha-beta pruning), constraint satisfaction problems (backtracking search with constraint propagation), and an introduction to learning methods including identification trees and neural nets — the last of which overlaps deliberately with 6.036. The course's organizing idea is that "intelligent behavior" in a huge range of problems reduces to *search over a state space*, and the differences between search algorithms are really differences in what extra information (a heuristic, an adversary, a set of constraints) is exploited to search less of that space.

## Why it matters for this portfolio

This is the course that reframes graph algorithms from 6.006 as instances of a more general search paradigm:
- A* is literally Dijkstra's algorithm plus a heuristic function, and its optimality proof depends on the heuristic being *admissible* — a condition this folder's Applied-Theory script states and tests directly.
- Alpha-beta pruning's correctness proof (that pruned branches provably cannot affect the minimax value) is the same kind of "safe to skip" argument used to justify greedy choices in Dijkstra (6.006).
- Constraint satisfaction with arc consistency generalizes the graph-coloring problem introduced in 6.042's graph theory unit into a full search-plus-propagation framework.

## What I focused on

The `Applied-Theory/` script implements three classical AI algorithms from scratch: (1) A* search on a grid with an obstacle field, using Euclidean distance as an admissible heuristic, verified against plain Dijkstra to confirm both find the same optimal path length; (2) minimax with alpha-beta pruning on a small combinatorial game, verified to produce the identical game-value decision as plain minimax while visiting provably fewer nodes; and (3) a backtracking constraint-satisfaction solver with forward checking, applied to graph coloring, benchmarked against naive backtracking without propagation.

## Folder contents

- [`Psets/pset_roadmap.md`](./Psets/pset_roadmap.md) — topic-by-topic syllabus breakdown.
- [`Applied-Theory/search_and_csp.py`](./Applied-Theory/search_and_csp.py) — A* with an admissible-heuristic check against Dijkstra, minimax with alpha-beta pruning benchmarked for node-count reduction, and a forward-checking CSP solver for graph coloring.
