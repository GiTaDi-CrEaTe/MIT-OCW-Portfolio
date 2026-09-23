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

def build_grid_graph(width, height, obstacles):
    """Grid graph: nodes are (x, y) cells not in `obstacles`; 4-connected,
    unit edge cost."""
    nodes = set()
    for x in range(width):
        for y in range(height):
            if (x, y) not in obstacles:
                nodes.add((x, y))

    def neighbors(cell):
        x, y = cell
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nxt = (x + dx, y + dy)
            if nxt in nodes:
                yield nxt

    return nodes, neighbors


def euclidean_heuristic(a, b):
    """
    h(n) = straight-line distance from n to the goal.
    Admissibility claim: on a 4-connected unit-cost grid, the true remaining
    cost from any cell to the goal is at least the Euclidean distance between
    them (a straight line is the shortest possible path in the continuous
    relaxation of the problem; the grid can only make the actual path longer
    by forcing detours around obstacles or axis-aligned moves). So
    h(n) <= true_cost(n, goal) always -- h never overestimates, which is
    exactly the admissibility condition A*'s optimality proof requires.
    """
    return math.dist(a, b)


def dijkstra_grid(nodes, neighbors, start, goal):
    """Uniform-cost search (Dijkstra) as a ground-truth baseline: guaranteed
    optimal, but explores purely by accumulated cost with no goal-directed
    guidance."""
    dist = {start: 0}
    prev = {}
    visited = set()
    heap = [(0, start)]
    expansions = 0
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        expansions += 1
        if node == goal:
            break
        for nxt in neighbors(node):
            nd = d + 1
            if nxt not in dist or nd < dist[nxt]:
                dist[nxt] = nd
                prev[nxt] = node
                heapq.heappush(heap, (nd, nxt))
    return dist.get(goal, math.inf), expansions


