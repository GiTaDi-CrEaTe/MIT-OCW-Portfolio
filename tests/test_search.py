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


