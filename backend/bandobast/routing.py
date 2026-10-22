"""Reachability and travel-time layer.

Custom A* over a weighted road graph with blocked-edge handling and an
admissible haversine/max-speed heuristic. Expanded-node count and frontier
size are exposed (proposal: "the algorithm itself inspectable"), and every
result is cross-checked against a Dijkstra ground truth.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from .domain import RoadGraph, RouteResult

EARTH_RADIUS_KM = 6371.0


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km between (lat, lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def edge_time_min(length_m: float, speed_kmph: float) -> float:
    """Travel time in minutes for an edge."""
    return (length_m / 1000.0) / max(speed_kmph, 1e-6) * 60.0


@dataclass
class Network:
    """Adjacency representation built once per scenario and reused for queries."""

    coords: dict[int, tuple[float, float]]
    # node -> list of (neighbour, time_min)
    adj: dict[int, list[tuple[int, float]]] = field(default_factory=dict)

    @classmethod
    def from_graph(cls, graph: RoadGraph) -> Network:
        adj: dict[int, list[tuple[int, float]]] = {n: [] for n in graph.coords}
        for e in graph.edges:
            if e.blocked:
                continue  # blocked segments are removed, not penalised
            t = edge_time_min(e.length_m, e.speed_kmph)
            adj.setdefault(e.u, []).append((e.v, t))
            adj.setdefault(e.v, []).append((e.u, t))  # undirected roads
        return cls(coords=dict(graph.coords), adj=adj)

    def heuristic_min(self, node: int, goal: int, max_speed_kmph: float) -> float:
        """Admissible: straight-line distance at the max possible speed is a
        lower bound on any real travel time, so A* stays optimal."""
        km = haversine_km(self.coords[node], self.coords[goal])
        return km / max(max_speed_kmph, 1e-6) * 60.0


def astar(net: Network, source: int, target: int, max_speed_kmph: float) -> RouteResult:
    """A* on travel-time edges. Reports expansion stats and the path."""
    if source not in net.adj or target not in net.adj:
        return RouteResult(source=source, target=target, reachable=False)

    g: dict[int, float] = {source: 0.0}
    came_from: dict[int, int] = {}
    counter = 0  # tie-breaker keeps the heap deterministic
    start_h = net.heuristic_min(source, target, max_speed_kmph)
    open_heap: list[tuple[float, int, int]] = [(start_h, counter, source)]
    closed: set[int] = set()
    expanded = 0
    max_frontier = 1

    while open_heap:
        max_frontier = max(max_frontier, len(open_heap))
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == target:
            path = _reconstruct(came_from, current)
            return RouteResult(
                source=source, target=target, reachable=True,
                eta_min=g[current], cost_min=g[current], path=path,
                expanded_nodes=expanded, max_frontier=max_frontier,
            )
        closed.add(current)
        expanded += 1
        for nbr, w in net.adj.get(current, []):
            if nbr in closed:
                continue
            tentative = g[current] + w
            if tentative < g.get(nbr, math.inf):
                g[nbr] = tentative
                came_from[nbr] = current
                counter += 1
                f = tentative + net.heuristic_min(nbr, target, max_speed_kmph)
                heapq.heappush(open_heap, (f, counter, nbr))

    return RouteResult(
        source=source, target=target, reachable=False,
        expanded_nodes=expanded, max_frontier=max_frontier,
    )


def dijkstra_search(net: Network, source: int, target: int) -> tuple[float | None, int]:
    """Ground-truth shortest travel time (minutes) and the number of nodes
    expanded. Dijkstra is A* with h=0, so it expands outward in every direction
    and settles more nodes than A* for the same optimal cost -- the classic
    informed-vs-uninformed search trade-off."""
    dist: dict[int, float] = {source: 0.0}
    counter = 0
    heap: list[tuple[float, int, int]] = [(0.0, counter, source)]
    seen: set[int] = set()
    expanded = 0
    while heap:
        d, _, node = heapq.heappop(heap)
        if node in seen:
            continue
        if node == target:
            return d, expanded
        seen.add(node)
        expanded += 1
        for nbr, w in net.adj.get(node, []):
            nd = d + w
            if nd < dist.get(nbr, math.inf):
                dist[nbr] = nd
                counter += 1
                heapq.heappush(heap, (nd, counter, nbr))
    return None, expanded


def dijkstra_cost(net: Network, source: int, target: int) -> float | None:
    """Ground-truth shortest travel time (minutes), or None if unreachable."""
    return dijkstra_search(net, source, target)[0]


def route_with_groundtruth(
    net: Network, source: int, target: int, max_speed_kmph: float
) -> RouteResult:
    """A* result annotated with Dijkstra's cost and expansion count, for the
    correctness check and the efficiency comparison."""
    r = astar(net, source, target, max_speed_kmph)
    r.dijkstra_cost_min, r.dijkstra_expanded = dijkstra_search(net, source, target)
    return r


def _reconstruct(came_from: dict[int, int], node: int) -> list[int]:
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path
