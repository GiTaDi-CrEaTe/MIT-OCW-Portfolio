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


def minimax_alpha_beta(node, maximizing, alpha, beta, counter):
    """
    Minimax with alpha-beta pruning.

    Correctness claim: pruning a branch never changes the value returned at
    the root. Proof sketch for the MAX case: suppose at a MAX node we have
    already found a child value >= beta (the best value the MIN ancestor
    above us is guaranteed to be able to force elsewhere). Any further
    children of this MAX node can only make this node's value larger still
    (MAX only ever increases its choice), which the MIN ancestor will never
    select anyway once it has an alternative <= beta available. So the
    remaining children are provably irrelevant to the final root value and
    can be skipped -- this is exactly a beta cutoff. The alpha cutoff at MIN
    nodes is the mirror-image argument.
    """
    counter[0] += 1
    if not node.children:
        return node.value

    if maximizing:
        value = -math.inf
        for child in node.children:
            value = max(value, minimax_alpha_beta(child, False, alpha, beta, counter))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # beta cutoff: remaining siblings cannot affect the result
        return value
    else:
        value = math.inf
        for child in node.children:
            value = min(value, minimax_alpha_beta(child, True, alpha, beta, counter))
            beta = min(beta, value)
            if alpha >= beta:
                break  # alpha cutoff
        return value


def build_random_game_tree(depth, branching, rng):
    """Builds a synthetic game tree of the given depth/branching factor with
    random leaf evaluation values, for benchmarking."""
    if depth == 0:
        return GameNode(value=rng.randint(-100, 100))
    children = [build_random_game_tree(depth - 1, branching, rng) for _ in range(branching)]
    return GameNode(children=children)


# ===========================================================================
# PART 3  --  Backtracking CSP with forward checking, applied to graph coloring
# (Pset 6-7)
# ===========================================================================

def backtracking_naive(variables, domains, constraints, assignment=None, counter=None):
    """Chronological backtracking with NO propagation: assign a value,
    recurse, and only check consistency of the CURRENT variable against
    already-assigned neighbors."""
    if assignment is None:
        assignment = {}
    if counter is None:
        counter = [0]
    counter[0] += 1

    if len(assignment) == len(variables):
        return dict(assignment), counter[0]

    unassigned = [v for v in variables if v not in assignment]
    var = unassigned[0]

    for value in domains[var]:
        if all(other not in assignment or assignment[other] != value
               for other in constraints.get(var, [])):
            assignment[var] = value
            result, _ = backtracking_naive(variables, domains, constraints, assignment, counter)
            if result is not None:
                return result, counter[0]
            del assignment[var]

    return None, counter[0]


def backtracking_forward_checking(variables, domains, constraints,
                                   assignment=None, local_domains=None, counter=None):
    """
    Backtracking with forward checking: whenever a variable is assigned,
    immediately REMOVE the assigned value from the domains of all its
    not-yet-assigned neighbors. If any neighbor's domain becomes empty, this
    branch is guaranteed to fail (no legal value remains for that neighbor)
    and can be pruned immediately -- without waiting to assign that neighbor
    and discover the failure later. This is strictly more informed than
    naive backtracking, which only detects the same failure once it actually
    tries (and fails) every value for that neighbor.
    """
    if assignment is None:
        assignment = {}
    if local_domains is None:
        local_domains = {v: list(domains[v]) for v in variables}
    if counter is None:
        counter = [0]
    counter[0] += 1

    if len(assignment) == len(variables):
        return dict(assignment), counter[0]

    unassigned = [v for v in variables if v not in assignment]
    var = unassigned[0]

    for value in list(local_domains[var]):
        assignment[var] = value

        # Forward checking step: prune this value from neighbors' domains.
        removed = []
        domain_wipeout = False
        for neighbor in constraints.get(var, []):
            if neighbor not in assignment and value in local_domains[neighbor]:
                local_domains[neighbor].remove(value)
                removed.append(neighbor)
                if not local_domains[neighbor]:
                    domain_wipeout = True

        if not domain_wipeout:
            result, _ = backtracking_forward_checking(
                variables, domains, constraints, assignment, local_domains, counter)
            if result is not None:
                return result, counter[0]

        # Undo forward-checking pruning before trying the next value.
        for neighbor in removed:
            local_domains[neighbor].append(value)
        del assignment[var]

    return None, counter[0]


def build_neighbor_constraints(edges):
    constraints = {}
    for u, v in edges:
        constraints.setdefault(u, []).append(v)
        constraints.setdefault(v, []).append(u)
    return constraints


# ===========================================================================
# Self-verification
# ===========================================================================

