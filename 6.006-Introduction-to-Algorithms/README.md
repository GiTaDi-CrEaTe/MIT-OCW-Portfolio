# 6.006 — Introduction to Algorithms

**MIT OCW subject:** 6.006, EECS.

## What this course is

6.006 covers the standard rigorous algorithms sequence: asymptotic analysis, sorting, hashing, binary search trees and balanced trees (AVL), graph representations and traversal (BFS/DFS), shortest paths (Dijkstra, Bellman-Ford), and an introduction to dynamic programming. The course's discipline is proof-driven — every algorithm comes with a correctness argument (usually a loop invariant or an exchange argument) and a formal running-time bound, not just an implementation.

## Why it matters for this portfolio

This course is where 6.042's induction and graph theory become executable, and where the asymptotic vocabulary (O, Θ, Ω) that gets used loosely in 6.036 and 6.034 gets its formal definition:
- The AVL tree's rebalancing correctness proof is structural induction on tree height (6.042, Pset 3-4).
- Dijkstra's correctness proof is an inductive argument over the order in which nodes are finalized.
- Hashing's expected-case analysis leans on linearity of expectation (6.042, Pset 14) applied to collision-counting.

## What I focused on

The `Applied-Theory/` script builds two structurally different but philosophically related things from scratch: (1) a self-balancing AVL binary search tree, whose entire value proposition is a *proof* — that height stays O(log n) under arbitrary insertion order — made concrete via empirical height tracking against a naive unbalanced BST; and (2) a from-scratch graph library implementing BFS, DFS, and Dijkstra's algorithm on an adjacency-list representation, with an empirical runtime-scaling experiment that checks the claimed asymptotic complexity against wall-clock behavior.

## Folder contents

- [`Psets/pset_roadmap.md`](./Psets/pset_roadmap.md) — topic-by-topic syllabus breakdown.
- [`Applied-Theory/graph_algorithms_and_data_structures.py`](./Applied-Theory/graph_algorithms_and_data_structures.py) — AVL tree with rotation-based rebalancing, BFS/DFS/Dijkstra graph algorithms, and an empirical complexity-scaling benchmark.
