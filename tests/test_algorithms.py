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


def test_bfs_shortest_paths():
    if hasattr(algo, "Graph"):
        g = algo.Graph()
        edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (1, 5)]
        for u, v in edges:
            g.add_edge(u, v)
            g.add_edge(v, u)
        order, distance = g.bfs(0)
        assert distance[0] == 0
        assert distance[5] == 2  # 0 -> 1 -> 5
    elif hasattr(algo, "bfs"):
        graph = {0: [(1,), (2,)], 1: [(0,), (3,), (4,)], 2: [(0,), (4,)], 3: [(1,), (5,)], 4: [(1,), (2,), (5,)], 5: [(3,), (4,)]}
        dist, _ = algo.bfs(graph, 0)
        assert dist[0] == 0
        assert dist[5] == 3


def test_dijkstra_correctness():
    if hasattr(algo, "Graph"):
        wg = algo.Graph()
        wg.add_edge(0, 1, 4)
        wg.add_edge(0, 2, 1)
        wg.add_edge(2, 1, 2)
        wg.add_edge(1, 3, 1)
        wg.add_edge(2, 3, 5)
        wg.add_edge(3, 4, 3)
        dist = wg.dijkstra(0)
        assert dist[0] == 0
        assert dist[1] == 3  # 0 -> 2 -> 1
        assert dist[2] == 1
        assert dist[3] == 4
        assert dist[4] == 7
    elif hasattr(algo, "dijkstra"):
        graph = {0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: [(4, 3)], 4: []}
        dist, _ = algo.dijkstra(graph, 0)
        assert dist[4] == 7



def test_dijkstra_rejects_negative_weights():
    """Dijkstra should raise ValueError on negative edge weights."""
    if hasattr(algo, "Graph"):
        g = algo.Graph()
        g.add_edge(0, 1, -5)
        g.add_edge(1, 2, 3)
        with pytest.raises(ValueError, match="non-negative"):
            g.dijkstra(0)


def test_avl_tree_no_duplicates():
    """Inserting duplicate keys should not increase the tree."""
    if hasattr(algo, "avl_insert"):
        root = None
        for k in [5, 3, 7, 5, 3, 7]:
            root = algo.avl_insert(root, k)
        out = []
        algo.avl_inorder(root, out)
        assert out == [3, 5, 7]
    elif hasattr(algo, "AVLTree"):
        tree = algo.AVLTree()
        for k in [5, 3, 7, 5, 3, 7]:
            tree.insert(k)
        assert tree.inorder() == [3, 5, 7]


def test_bfs_single_node():
    """BFS from a node with no edges should return just that node."""
    if hasattr(algo, "Graph"):
        g = algo.Graph()
        g.add_node(42)
        order, distance = g.bfs(42)
        assert order == [42]
        assert distance == {42: 0}


def test_dfs_visits_all_reachable():
    """DFS should visit all reachable nodes."""
    if hasattr(algo, "Graph"):
        g = algo.Graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        order = g.dfs(0)
        assert set(order) == {0, 1, 2, 3}
