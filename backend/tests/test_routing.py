import math

from bandobast.domain import Edge, RoadGraph
from bandobast.routing import Network, astar, dijkstra_cost, haversine_km
from bandobast.synth import build_grid_graph


def test_astar_matches_dijkstra_on_grid():
    net = Network.from_graph(build_grid_graph(seed=7))
    nodes = list(net.coords)
    checked = 0
    for s in nodes[:6]:
        for t in nodes:
            r = astar(net, s, t, max_speed_kmph=60)
            d = dijkstra_cost(net, s, t)
            if r.reachable:
                assert d is not None
                assert math.isclose(r.cost_min, d, rel_tol=1e-9, abs_tol=1e-9)
                checked += 1
            else:
                assert d is None
    assert checked > 0


def test_heuristic_admissible():
    net = Network.from_graph(build_grid_graph(seed=3))
    nodes = list(net.coords)
    for s in nodes[:5]:
        for t in nodes:
            r = astar(net, s, t, max_speed_kmph=60)
            if r.reachable:
                h = net.heuristic_min(s, t, 60)
                # Admissible heuristic never overestimates true cost.
                assert h <= r.cost_min + 1e-9


def test_blocked_edge_removes_path():
    coords = {0: (0.0, 0.0), 1: (0.0, 0.01)}
    length = haversine_km(coords[0], coords[1]) * 1000.0
    open_graph = RoadGraph(coords=coords, edges=[Edge(u=0, v=1, length_m=length, speed_kmph=40)])
    blocked = RoadGraph(coords=coords,
                        edges=[Edge(u=0, v=1, length_m=length, speed_kmph=40, blocked=True)])
    assert astar(Network.from_graph(open_graph), 0, 1, 60).reachable
    assert not astar(Network.from_graph(blocked), 0, 1, 60).reachable


def test_expansion_stats_reported():
    net = Network.from_graph(build_grid_graph(seed=1))
    r = astar(net, 0, 35, max_speed_kmph=60)
    assert r.reachable
    assert r.expanded_nodes > 0
    assert r.max_frontier >= 1
