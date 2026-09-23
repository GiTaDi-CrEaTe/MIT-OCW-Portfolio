"""
6.006 Applied Theory  --  Balanced Trees and Graph Algorithms from First Principles
==================================================================================

Two self-contained pieces, both built with plain Python (no `collections`
tricks beyond `deque`, no external graph/tree libraries):

  1. An AVL self-balancing binary search tree, proving empirically what the
     O(log n) height invariant guarantees in theory.

  2. A from-scratch adjacency-list graph with BFS, DFS, and Dijkstra's
     algorithm, closing with an empirical runtime-scaling benchmark that
     checks measured wall-clock growth against the claimed asymptotic bounds.
"""

import random
import time
from collections import deque
import heapq


# ===========================================================================
# PART 1  --  AVL Tree (Pset 4-5)
# ===========================================================================

