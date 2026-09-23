"""
Foundations Lab  --  Capstone Experiment 3: Heuristic Search Efficiency and Optimality
==================================================================================

Research Question:
  How much search can admissible heuristic information eliminate without sacrificing
  path optimality, and what happens when the heuristic overestimates?

Theoretical Guarantees:
  1. Dijkstra Algorithm (h = 0):
     Guaranteed optimal on non-negative edge weights. Explores a circular wavefront
     of area O(V).
  2. A* with Admissible and Consistent Heuristic (h(n) <= h*(n)):
     Guaranteed optimal: the first time a goal node is expanded, its path is minimal.
     Prunes the search space by focusing the expansion ellipse toward the goal.
     Since Manhattan distance h_M(u, v) >= Euclidean distance h_E(u, v) on a 4-connected grid,
     h_M strictly dominates h_E: A*(h_M) expands no more nodes than A*(h_E).
  3. Lexicographic Tie-Breaking (Preserving Admissibility):
     On grids, many paths share identical f-scores f = g + h. Rather than inflating h
     by a factor (1 + eps) which breaks strict admissibility (since (1 + eps)*h can exceed h*),
     true lexicographic tie-breaking keeps f = g + h strictly unscaled, and breaks ties
     among equal-f states by prioritizing states with smaller remaining heuristic distance h.
     This preserves the mathematical admissibility theorem while collapsing plateau expansions.
  4. A* with Inadmissible Heuristic (weight w > 1.0, e.g., w * h_M):
     Sacrifices the mathematical optimality guarantee. It expands significantly fewer nodes
     ("greedy search"), but risks returning suboptimal paths.
"""

import heapq
import math
from typing import Callable, Dict, List, Optional, Set, Tuple
import numpy as np


class GridMap:
    """2D grid with passable (0) and blocked (1) cells."""
    def __init__(self, width: int, height: int, obstacle_density: float, seed: int):
        self.width = width
        self.height = height
        rng = np.random.default_rng(seed)
        self.grid = (rng.random((height, width)) < obstacle_density).astype(int)
        # Guarantee start and goal are open
        self.grid[0, 0] = 0
        self.grid[height - 1, width - 1] = 0

    def neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = node
        nbrs = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] == 0:
                nbrs.append((nx, ny))
        return nbrs


def h_zero(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return 0.0


def h_euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def h_manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def h_manhattan_tiebreak(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Maintained for backward compatibility: pure Manhattan distance.
    Tie-breaking is now handled lexicographically in the priority queue.
    """
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def h_inadmissible(a: Tuple[int, int], b: Tuple[int, int], weight: float = 1.5) -> float:
    """Overestimating heuristic: breaks admissibility to demonstrate optimality loss."""
    return weight * float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def run_astar(
    grid: GridMap,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic_fn: Callable[[Tuple[int, int], Tuple[int, int]], float],
    tie_break: bool = False,
) -> Tuple[Optional[float], int]:
    """
    Runs A* search with optional lexicographic tie-breaking.
    When tie_break is True, equal f-score states are resolved by preferring
    the state with minimal remaining heuristic distance h. Because f = g + h
    is strictly unmodified and h is unscaled, admissibility and optimality
    guarantees are fully preserved.

    Returns: (path_length, nodes_expanded). Returns (None, nodes_expanded) if unreachable.
    """
    counter = 0  # Tie-breaker for heap on identical (f, secondary)
    pq: List[Tuple[float, float, int, Tuple[int, int]]] = []  # (f, secondary_key, count, node)
    h_start = heuristic_fn(start, goal)
    sec_start = h_start if tie_break else 0.0
    heapq.heappush(pq, (h_start, sec_start, counter, start))

    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    closed_set: Set[Tuple[int, int]] = set()
    nodes_expanded = 0

    while pq:
        f, _, _, curr = heapq.heappop(pq)

        if curr in closed_set:
            continue
        closed_set.add(curr)
        nodes_expanded += 1

        if curr == goal:
            return g_score[goal], nodes_expanded

        for nbr in grid.neighbors(curr):
            tentative_g = g_score[curr] + 1.0
            if nbr not in g_score or tentative_g < g_score[nbr]:
                g_score[nbr] = tentative_g
                h_nbr = heuristic_fn(nbr, goal)
                sec_nbr = h_nbr if tie_break else 0.0
                counter += 1
                heapq.heappush(pq, (tentative_g + h_nbr, sec_nbr, counter, nbr))

    return None, nodes_expanded


def run_benchmark_sweep(
    grid_sizes: List[int] = [30, 50, 70],
    densities: List[float] = [0.0, 0.1, 0.2, 0.3],
    trials_per_config: int = 15,
) -> Dict[str, Dict]:
    """
    Executes a multi-parameter sweep across grid dimensions and obstacle densities.
    """
    algorithms = {
        "Dijkstra (h=0)": (h_zero, False),
        "A* (Euclidean)": (h_euclidean, False),
        "A* (Manhattan)": (h_manhattan, False),
        "A* (Manhattan + Lexicographic Tie-Break)": (h_manhattan, True),
        "A* (Inadmissible w=1.5)": (lambda a, b: h_inadmissible(a, b, weight=1.5), False),
    }

    results: Dict[str, Dict] = {algo: {"nodes": [], "suboptimal_count": 0, "total_valid": 0} for algo in algorithms}
    sweep_records = []

    for size in grid_sizes:
        start = (0, 0)
        goal = (size - 1, size - 1)
        for density in densities:
            nodes_by_algo: Dict[str, List[int]] = {k: [] for k in algorithms}
            cost_by_algo: Dict[str, List[float]] = {k: [] for k in algorithms}

            valid_trials = 0
            seed = 42
            while valid_trials < trials_per_config:
                grid = GridMap(size, size, density, seed=seed)
                seed += 1

                # First run Dijkstra to get ground-truth shortest path
                opt_cost, dij_nodes = run_astar(grid, start, goal, algorithms["Dijkstra (h=0)"][0], tie_break=False)
                if opt_cost is None:
                    continue  # Unreachable configuration, discard

                valid_trials += 1
                nodes_by_algo["Dijkstra (h=0)"].append(dij_nodes)
                cost_by_algo["Dijkstra (h=0)"].append(opt_cost)

                for name, (fn, tb) in algorithms.items():
                    if name == "Dijkstra (h=0)":
                        continue
                    cost, nodes = run_astar(grid, start, goal, fn, tie_break=tb)
                    nodes_by_algo[name].append(nodes)
                    cost_by_algo[name].append(cost if cost is not None else float("inf"))

                    if cost is not None and cost > opt_cost + 1e-6:
                        results[name]["suboptimal_count"] += 1
                    results[name]["total_valid"] += 1

            record = {
                "size": size,
                "density": density,
                "dijkstra_nodes": float(np.mean(nodes_by_algo["Dijkstra (h=0)"])),
                "dijkstra_nodes_std": float(np.std(nodes_by_algo["Dijkstra (h=0)"])),
                "euclidean_nodes": float(np.mean(nodes_by_algo["A* (Euclidean)"])),
                "manhattan_nodes": float(np.mean(nodes_by_algo["A* (Manhattan)"])),
                "tiebreak_nodes": float(np.mean(nodes_by_algo["A* (Manhattan + Lexicographic Tie-Break)"])),
                "tiebreak_nodes_std": float(np.std(nodes_by_algo["A* (Manhattan + Lexicographic Tie-Break)"])),
                "inadmissible_nodes": float(np.mean(nodes_by_algo["A* (Inadmissible w=1.5)"])),
                "manhattan_reduction_pct": float(
                    (1.0 - np.mean(nodes_by_algo["A* (Manhattan)"]) / np.mean(nodes_by_algo["Dijkstra (h=0)"])) * 100.0
                ),
                "tiebreak_reduction_pct": float(
                    (1.0 - np.mean(nodes_by_algo["A* (Manhattan + Lexicographic Tie-Break)"]) / np.mean(nodes_by_algo["Dijkstra (h=0)"])) * 100.0
                ),
            }
            sweep_records.append(record)

    return {"records": sweep_records, "summary": results}


if __name__ == "__main__":
    print("=" * 86)
    print("EXPERIMENT 3: A* Heuristic Efficiency vs Dijkstra Optimality Benchmark")
    print("=" * 86)
    data = run_benchmark_sweep(grid_sizes=[30, 50], densities=[0.0, 0.1, 0.2, 0.25], trials_per_config=10)

    print(f"{'Size':>5} | {'Density':>7} | {'Dijkstra':>9} | {'Euclidean':>10} | {'Manhattan':>10} | {'Tie-Break':>10} | {'Inadm(1.5x)':>11} | {'Tie Savings':>11}")
    print("-" * 86)
    for r in data["records"]:
        print(
            f"{r['size']:5d} | {r['density']:7.2f} | {r['dijkstra_nodes']:9.1f} | "
            f"{r['euclidean_nodes']:10.1f} | {r['manhattan_nodes']:10.1f} | "
            f"{r['tiebreak_nodes']:10.1f} | {r['inadmissible_nodes']:11.1f} | {r['tiebreak_reduction_pct']:10.1f}%"
        )

    print("\n" + "=" * 86)
    print("Optimality Verification:")
    for algo, res in data["summary"].items():
        if res["total_valid"] > 0:
            rate = (res["suboptimal_count"] / res["total_valid"]) * 100.0
            print(f"  {algo:<40}: Suboptimal paths = {res['suboptimal_count']}/{res['total_valid']} ({rate:.1f}%)")
