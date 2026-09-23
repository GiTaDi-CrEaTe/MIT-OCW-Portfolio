"""
Tests for 6.006 Algorithms & Data Structures
"""

import random
import pytest
from conftest import load_course_module

algo = load_course_module("algo_scratch", "6.006-Introduction-to-Algorithms/Applied-Theory/graph_algorithms_and_data_structures.py")


def test_avl_tree_balance_and_ordering():
    rng = random.Random(6006)
    keys = list(range(1, 101))
    rng.shuffle(keys)

    if hasattr(algo, "avl_insert"):
        root = None
        for k in keys:
            root = algo.avl_insert(root, k)
        out = []
        algo.avl_inorder(root, out)
        assert out == list(range(1, 101))
        h = algo._height(root)
        assert h <= 10
    elif hasattr(algo, "AVLTree"):
        tree = algo.AVLTree()
        for k in keys:
            tree.insert(k)
        assert tree.verify_avl() is True
        assert tree.inorder() == list(range(1, 101))


