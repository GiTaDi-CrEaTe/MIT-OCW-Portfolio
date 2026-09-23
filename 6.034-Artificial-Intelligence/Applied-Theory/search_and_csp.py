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


def a_star_grid(nodes, neighbors, start, goal, heuristic):
    """
    A* search: identical to Dijkstra except nodes in the priority queue are
    ordered by f(n) = g(n) + h(n), where g(n) is the accumulated cost so far
    and h(n) is the admissible heuristic estimate of remaining cost.

    Optimality proof sketch (why this still finds the shortest path):
    because h never overestimates, f(n) never overestimates the true cost of
    the best path through n. So when the goal is popped from the priority
    queue, no other node in the queue could possibly lead to a shorter path
    to the goal -- if one did, its f-value (a valid lower bound on its true
    cost) would have been smaller and it would have been popped first.
    """
    g_score = {start: 0}
    prev = {}
    visited = set()
    heap = [(heuristic(start, goal), start)]
    expansions = 0
    while heap:
        f, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        expansions += 1
        if node == goal:
            break
        for nxt in neighbors(node):
            tentative_g = g_score[node] + 1
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                prev[nxt] = node
                heapq.heappush(heap, (tentative_g + heuristic(nxt, goal), nxt))
    return g_score.get(goal, math.inf), expansions


# ===========================================================================
# PART 2  --  Minimax with alpha-beta pruning (Pset 4-5)
# ===========================================================================

class GameNode:
    """A minimal synthetic game tree node: internal nodes alternate
    MAX/MIN; leaves carry a static evaluation value."""

    def __init__(self, children=None, value=None):
        self.children = children or []
        self.value = value  # only set for leaves


def minimax(node, maximizing, counter):
    """Plain minimax with no pruning -- the ground-truth baseline. `counter`
    is a mutable [int] used to count node visits for the benchmark."""
    counter[0] += 1
    if not node.children:
        return node.value
    if maximizing:
        return max(minimax(child, False, counter) for child in node.children)
    else:
        return min(minimax(child, True, counter) for child in node.children)


