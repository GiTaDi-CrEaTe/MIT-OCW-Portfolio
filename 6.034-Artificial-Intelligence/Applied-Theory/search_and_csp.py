"""
6.034 Applied Theory  --  Search, Adversarial Reasoning, and CSPs from Scratch
==============================================================================

Three classical AI algorithms, each implemented from scratch and each
verified against a slower/naive baseline to make a specific theoretical
claim concrete rather than assumed:

  1. A* search on a grid, with an ADMISSIBLE heuristic (Euclidean distance,
     which never overestimates true remaining cost) -- verified to find a
     path of the exact same optimal length as plain Dijkstra, while
     expanding far fewer nodes.

  2. Minimax with alpha-beta pruning -- verified to return the IDENTICAL
     game value as plain minimax on the same tree, while visiting
     provably fewer nodes (the pruning-correctness claim made concrete).

  3. Backtracking CSP solver with forward checking, applied to graph
     coloring -- verified against naive backtracking to find valid
     colorings with far fewer nodes explored.

Only the Python standard library (`heapq`, `math`) is used.
"""

import heapq
import math
import random


# ===========================================================================
# PART 1  --  A* search vs. Dijkstra on a grid with obstacles (Pset 2-3)
# ===========================================================================

