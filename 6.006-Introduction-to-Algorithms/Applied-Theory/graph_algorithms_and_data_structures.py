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


def _rotate_right(y):
    """
    Standard AVL right rotation. Used when the left subtree is too tall.
    Correctness: this is a local re-wiring that preserves the BST property
    (in-order traversal is unchanged) while shifting height from the left
    subtree to the right subtree.
    """
    x = y.left
    T2 = x.right
    x.right = y
    y.left = T2
    _update_height(y)
    _update_height(x)
    return x  # new subtree root


def _rotate_left(x):
    """Mirror image of _rotate_right."""
    y = x.right
    T2 = y.left
    y.left = x
    x.right = T2
    _update_height(x)
    _update_height(y)
    return y


def avl_insert(node, key):
    """
    Standard BST insertion, followed by rebalancing on the way back up the
    recursion. The AVL invariant maintained after every insertion:
        |balance_factor(node)| <= 1   for every node
    This is what guarantees height = O(log n): a tree satisfying this
    invariant is provably bounded in height by ~1.44 log2(n) (a fact tied to
    the Fibonacci recurrence -- the minimal-node AVL tree of height h has
    exactly Fib(h+2) - 1 nodes), in sharp contrast to an unbalanced BST, which
    degrades to height n on sorted input.
    """
    if node is None:
        return AVLNode(key)
    if key < node.key:
        node.left = avl_insert(node.left, key)
    elif key > node.key:
        node.right = avl_insert(node.right, key)
    else:
        return node  # no duplicate keys

    _update_height(node)
    balance = _balance_factor(node)

    # Left-Left case
    if balance > 1 and key < node.left.key:
        return _rotate_right(node)
    # Right-Right case
    if balance < -1 and key > node.right.key:
        return _rotate_left(node)
    # Left-Right case
    if balance > 1 and key > node.left.key:
        node.left = _rotate_left(node.left)
        return _rotate_right(node)
    # Right-Left case
    if balance < -1 and key < node.right.key:
        node.right = _rotate_right(node.right)
        return _rotate_left(node)

    return node


def avl_inorder(node, out):
    if node:
        avl_inorder(node.left, out)
        out.append(node.key)
        avl_inorder(node.right, out)


