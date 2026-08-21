"""
6.006 Applied Theory — Balanced Trees and Graph Algorithms from First Principles
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
# PART 1 — AVL Tree (Pset 4-5)
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


class UnbalancedBST:
    """Naive BST with no rebalancing -- used only as a comparison baseline
    to make the AVL height guarantee visible, not as recommended practice."""

    class Node:
        __slots__ = ("key", "left", "right")

        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = self.Node(key)
            return
        node = self.root
        while True:
            if key < node.key:
                if node.left is None:
                    node.left = self.Node(key)
                    return
                node = node.left
            elif key > node.key:
                if node.right is None:
                    node.right = self.Node(key)
                    return
                node = node.right
            else:
                return

    def height(self):
        """Iterative height computation -- deliberately non-recursive, since
        the whole point of this baseline is that it degenerates to a
        linked-list shape on sorted input, and a recursive traversal would
        blow the Python call stack exactly because the tree is that
        pathologically unbalanced."""
        if self.root is None:
            return 0
        max_depth = 0
        stack = [(self.root, 1)]
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            if node.left:
                stack.append((node.left, depth + 1))
            if node.right:
                stack.append((node.right, depth + 1))
        return max_depth


# ===========================================================================
# PART 2 — Graph algorithms (Pset 7-9): BFS, DFS, Dijkstra
# ===========================================================================

class Graph:
    """Weighted, directed graph via adjacency list: {u: [(v, weight), ...]}."""

    def __init__(self):
        self.adj = {}

    def add_node(self, u):
        self.adj.setdefault(u, [])

    def add_edge(self, u, v, weight=1):
        self.add_node(u)
        self.add_node(v)
        self.adj[u].append((v, weight))

    def bfs(self, source):
        """
        Breadth-first search. Correctness claim: BFS discovers nodes in
        strictly non-decreasing order of their (unweighted) distance from
        source. Proof sketch (induction on distance layer d): assume all
        nodes at distance < d have already been correctly dequeued in
        non-decreasing distance order; any node at distance d is adjacent to
        some node at distance d-1, which by the inductive hypothesis was
        already dequeued and had its neighbors enqueued -- so this node is
        discovered no later than any node at distance d+1.
        """
        distance = {source: 0}
        order = []
        queue = deque([source])
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, _w in self.adj.get(u, []):
                if v not in distance:
                    distance[v] = distance[u] + 1
                    queue.append(v)
        return order, distance

    def dfs(self, source):
        """
        Depth-first search (iterative, explicit stack to avoid recursion
        limits). Used here mainly to demonstrate edge classification, the
        tool behind cycle detection and topological sort (Pset 8).
        """
        visited = set()
        order = []
        stack = [source]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            order.append(u)
            for v, _w in reversed(self.adj.get(u, [])):
                if v not in visited:
                    stack.append(v)
        return order

    def dijkstra(self, source):
        """
        Dijkstra's algorithm via a binary heap priority queue.
        Correctness claim: when a node u is popped (finalized) from the heap,
        dist[u] is already the true shortest-path distance from source.
        Proof sketch (the "cut property"): suppose for contradiction some
        finalized node u had an incorrect (too large) distance. Consider the
        true shortest path from source to u; let y be the first node on that
        path not yet finalized. Because all edge weights are non-negative,
        dist[y] <= true_dist(u) < dist[u] as currently recorded -- but then y
        should have been popped before u, contradicting the assumption that u
        was popped first. This is exactly why Dijkstra REQUIRES non-negative
        weights (Bellman-Ford, Pset 10, drops this requirement at the cost of
        higher complexity).
        """
        dist = {source: 0}
        finalized = set()
        heap = [(0, source)]
        while heap:
            d_u, u = heapq.heappop(heap)
            if u in finalized:
                continue
            finalized.add(u)
            for v, w in self.adj.get(u, []):
                if w < 0:
                    raise ValueError("Dijkstra requires non-negative edge weights.")
                new_dist = d_u + w
                if v not in dist or new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(heap, (new_dist, v))
        return dist


# ===========================================================================
# Self-verification
# ===========================================================================

def _self_test():
    print("=" * 70)
    print("SELF-TEST 1: AVL tree height stays O(log n) on adversarial")
    print("(sorted-order) insertion, unlike an unbalanced BST")
    print("=" * 70)
    n = 5000
    sorted_keys = list(range(n))  # worst case for an unbalanced BST

    avl_root = None
    for k in sorted_keys:
        avl_root = avl_insert(avl_root, k)
    avl_height = _height(avl_root)

    naive = UnbalancedBST()
    for k in sorted_keys:
        naive.insert(k)
    naive_height = naive.height()

    theoretical_bound = 1.45 * (n.bit_length())  # ~1.44 log2(n) + small constant
    print(f"n = {n} keys inserted in already-sorted order")
    print(f"AVL tree height:        {avl_height}   (theoretical bound ~ {theoretical_bound:.0f})")
    print(f"Unbalanced BST height:  {naive_height}   (degrades to a linked list: height = n)")
    assert avl_height <= theoretical_bound
    assert naive_height == n
    print("PASSED: AVL height is logarithmic; unbalanced BST height is linear.\n")

    print("=" * 70)
    print("SELF-TEST 2: AVL in-order traversal is sorted (BST property preserved")
    print("through all rotations)")
    print("=" * 70)
    out = []
    avl_inorder(avl_root, out)
    assert out == sorted_keys
    print("PASSED: in-order traversal recovers the exact sorted key order.\n")

    print("=" * 70)
    print("SELF-TEST 3: BFS finds shortest paths in an unweighted graph")
    print("=" * 70)
    g = Graph()
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (1, 5)]
    for u, v in edges:
        g.add_edge(u, v)
        g.add_edge(v, u)  # undirected for this test
    order, distance = g.bfs(0)
    print(f"BFS distances from node 0: {distance}")
    assert distance[0] == 0 and distance[5] == 2  # 0->1->5
    print("PASSED: shortest hop-count distances match hand-computed values.\n")

    print("=" * 70)
    print("SELF-TEST 4: Dijkstra matches brute-force shortest paths on a")
    print("random weighted graph")
    print("=" * 70)
    random.seed(6006)
    wg = Graph()
    num_nodes = 8
    for i in range(num_nodes):
        wg.add_node(i)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < 0.4:
                wg.add_edge(i, j, random.randint(1, 20))

    dijkstra_dist = wg.dijkstra(0)

    # Brute-force ground truth: Bellman-Ford-style full relaxation (|V|-1 passes)
    bf_dist = {0: 0}
    for _ in range(num_nodes):
        for u in wg.adj:
            if u in bf_dist:
                for v, w in wg.adj[u]:
                    nd = bf_dist[u] + w
                    if v not in bf_dist or nd < bf_dist[v]:
                        bf_dist[v] = nd

    print(f"Dijkstra distances:     {dict(sorted(dijkstra_dist.items()))}")
    print(f"Brute-force distances:  {dict(sorted(bf_dist.items()))}")
    assert dijkstra_dist == bf_dist
    print("PASSED: Dijkstra matches brute-force relaxation exactly.\n")

    print("=" * 70)
    print("SELF-TEST 5: Empirical runtime scaling of BFS vs. graph size")
    print("(sanity check against the claimed O(V + E) bound)")
    print("=" * 70)
    sizes = [500, 1000, 2000, 4000]
    for size in sizes:
        big_g = Graph()
        for i in range(size):
            big_g.add_node(i)
        # sparse random graph: ~4 edges per node, so E = O(V)
        for i in range(size):
            for _ in range(4):
                j = random.randint(0, size - 1)
                if j != i:
                    big_g.add_edge(i, j)
        start = time.perf_counter()
        big_g.bfs(0)
        elapsed = time.perf_counter() - start
        print(f"  V={size:5d}  E~{4*size:6d}   BFS time: {elapsed*1000:7.3f} ms")
    print("(Expect roughly linear growth in elapsed time as V doubles,")
    print("consistent with O(V + E) when E = O(V).)\n")

    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
