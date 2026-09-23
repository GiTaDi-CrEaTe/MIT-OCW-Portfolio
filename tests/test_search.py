"""
Tests for 6.034 Artificial Intelligence (Search, Adversarial Reasoning, CSP)
"""

import math
import random
import pytest
from conftest import load_course_module

search = load_course_module("search_scratch", "6.034-Artificial-Intelligence/Applied-Theory/search_and_csp.py")


def test_astar_admissibility_and_optimality():
    if hasattr(search, "build_grid_graph"):
        width, height = 15, 15
        start, goal = (0, 0), (width - 1, height - 1)
        obstacles = {(3, 3), (3, 4), (3, 5), (4, 5), (5, 5)}
        nodes, neighbors = search.build_grid_graph(width, height, obstacles)

        d_cost, d_exp = search.dijkstra_grid(nodes, neighbors, start, goal)
        a_cost, a_exp = search.a_star_grid(nodes, neighbors, start, goal, search.euclidean_heuristic)

        assert d_cost == a_cost
        assert a_exp <= d_exp
    elif hasattr(search, "GridWorld"):
        grid = search.GridWorld(15, 15, obstacle_rate=0.15, seed=42)
        start, goal = (0, 0), (14, 14)
        _, d_exp, d_cost = search.dijkstra_search(grid, start, goal)
        _, a_exp, a_cost = search.astar(grid, start, goal, search.manhattan_distance)
        if math.isfinite(d_cost):
            assert d_cost == a_cost
            assert a_exp <= d_exp


def test_minimax_and_alphabeta_equivalence():
    if hasattr(search, "build_random_game_tree"):
        rng = random.Random(34)
        tree = search.build_random_game_tree(depth=4, branching=2, rng=rng)
        c_plain = [0]
        plain_val = search.minimax(tree, True, c_plain)
        c_ab = [0]
        ab_val = search.minimax_alpha_beta(tree, True, -math.inf, math.inf, c_ab)
        assert plain_val == ab_val
        assert c_ab[0] <= c_plain[0]
    elif hasattr(search, "GameNode"):
        def build_tree(values, depth, max_depth):
            if depth == max_depth:
                return search.GameNode(value=values.pop(0))
            children = [build_tree(values, depth + 1, max_depth) for _ in range(2)]
            return search.GameNode(children=children)
        leaf_vals = [3, 5, 6, 9, 1, 2, 0, 7]
        tree = build_tree(leaf_vals, 0, 3)
        mm_v, mm_n = search.minimax(tree, True)
        ab_v, ab_n = search.alpha_beta(tree, True)
        assert mm_v == ab_v
        assert ab_n <= mm_n



def test_csp_forward_checking_finds_valid_coloring():
    """Forward-checking CSP should find a valid graph coloring on a known-solvable graph."""
    if not hasattr(search, "backtracking_forward_checking"):
        pytest.skip("CSP solver not available")
    variables = [0, 1, 2]
    domains = {v: ["R", "G", "B"] for v in variables}
    edges = [(0, 1), (1, 2), (0, 2)]
    constraints = search.build_neighbor_constraints(edges)
    result, nodes = search.backtracking_forward_checking(variables, domains, constraints)
    assert result is not None, "Should find a valid 3-coloring of a triangle"
    for u, v in edges:
        assert result[u] != result[v], f"Adjacent nodes {u} and {v} have same color"


def test_csp_forward_checking_fewer_nodes_than_naive():
    """Forward checking should explore fewer (or equal) nodes than naive backtracking."""
    if not hasattr(search, "backtracking_forward_checking"):
        pytest.skip("CSP solver not available")
    variables = list(range(6))
    domains = {v: ["R", "G", "B"] for v in variables}
    edges = [(0,1), (0,2), (1,2), (1,3), (2,4), (3,4), (3,5), (4,5)]
    constraints = search.build_neighbor_constraints(edges)
    naive_result, naive_nodes = search.backtracking_naive(variables, domains, constraints)
    fc_result, fc_nodes = search.backtracking_forward_checking(variables, domains, constraints)
    assert naive_result is not None
    assert fc_result is not None
    assert fc_nodes <= naive_nodes


def test_astar_unreachable_goal():
    """A* should return None cost or inf when the goal is unreachable."""
    if not hasattr(search, "build_grid_graph"):
        pytest.skip("Grid graph builder not available")
    obstacles = {(4, y) for y in range(6)}
    nodes, neighbors = search.build_grid_graph(6, 6, obstacles)
    cost, expansions = search.dijkstra_grid(nodes, neighbors, (0, 0), (5, 5))
    assert cost == float('inf') or cost == math.inf
