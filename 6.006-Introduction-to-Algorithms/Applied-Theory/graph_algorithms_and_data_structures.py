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

class AVLNode:
    __slots__ = ("key", "left", "right", "height")

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1  # height of a leaf is 1


def _height(node):
    return node.height if node else 0


def _balance_factor(node):
    return _height(node.left) - _height(node.right) if node else 0


def _update_height(node):
    node.height = 1 + max(_height(node.left), _height(node.right))


