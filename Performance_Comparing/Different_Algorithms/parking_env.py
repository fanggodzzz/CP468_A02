"""
Parking Environment Module

This module provides the environment setup, graph reading, traffic simulation,
and utility functions for the Q-learning parking lot recommendation system.
"""

import heapq
import os
import random

# ---------------------------------------------------------
# Constraints and weights
# ---------------------------------------------------------
W_MAX = 10
LAMBDA_WALK = 10.0
LAMBDA_PRICE = 1.0
PENALTY = 5000  # Important

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
PARKING_LOTS = []
INVALID_LOTS = []
LOT_INDEX = {}

p_avail = []
walk_km = []
price = []

graph = {}
traffic = {}

# Controlled dynamic environment, change by 50% chance
ENV_CHANGE_PROB = 0.5
script_dir = os.path.dirname(__file__)


# ---------------------------------------------------------
# Read graph
# ---------------------------------------------------------
def read_graph(filename="graph.txt"):
    """Read the graph from a file.

    Format:
        n m p in
        a1 a2 ... ap
        b1 b2 ... bin
        u1 v1 w1
        ...
    """
    global graph, PARKING_LOTS, INVALID_LOTS, LOT_INDEX, p_avail, walk_km, price

    graph = {}
    PARKING_LOTS = []
    INVALID_LOTS = []
    LOT_INDEX = {}
    p_avail = []
    walk_km = []
    price = []

    if not os.path.isabs(filename):
        filename = os.path.join(script_dir, filename)

    with open(filename, "r", encoding="utf-8") as f:
        first_line = f.readline()
        first_line = first_line.lstrip('\ufeff')
        header = first_line.strip().split()
        if len(header) < 4:
            raise ValueError("Graph file header must contain n, m, p, in")

        n, m, p_count, invalid_count = map(int, header[:4])
        parking_nodes = [int(x) for x in f.readline().strip().split() if x.strip()]
        invalid_nodes = [int(x) for x in f.readline().strip().split() if x.strip()]

        if len(parking_nodes) != p_count:
            raise ValueError(
                f"Graph file parking list length {len(parking_nodes)} does not match header p={p_count}"
            )

        if len(invalid_nodes) != invalid_count:
            raise ValueError(
                f"Graph file invalid list length {len(invalid_nodes)} does not match header in={invalid_count}"
            )

        if len(set(parking_nodes)) != len(parking_nodes):
            raise ValueError("Duplicate parking lot nodes are not allowed in the graph file")

        if len(set(invalid_nodes)) != len(invalid_nodes):
            raise ValueError("Duplicate invalid parking nodes are not allowed in the graph file")

        if not set(invalid_nodes).issubset(set(parking_nodes)):
            raise ValueError("Invalid parking lot nodes must be a subset of parking lot nodes")

        PARKING_LOTS = parking_nodes
        INVALID_LOTS = invalid_nodes
        LOT_INDEX = {lot: i for i, lot in enumerate(PARKING_LOTS)}
        p_avail = [1] * len(PARKING_LOTS)
        walk_km = [0] * len(PARKING_LOTS)
        price = [0] * len(PARKING_LOTS)

        for i in range(n):
            graph[i] = []

        for _ in range(m):
            line = f.readline().strip()
            if not line:
                continue
            u, v, w = map(int, line.split())
            graph[u].append((v, w))
            graph[v].append((u, w))


# ---------------------------------------------------------
# Traffic
# ---------------------------------------------------------
def traffic_multiplier(level):
    """Get the traffic multiplier based on traffic level."""
    if level == "low":
        return 1
    elif level == "medium":
        return 3
    else:
        return 7


def randomize_traffic():
    """Randomize the traffic for edges in the graph."""
    global traffic
    traffic = {}

    for u in graph:
        for v, _ in graph[u]:
            traffic[(u, v)] = random.choice(["low", "medium", "high"])


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
def create_environment(filename="graph.txt"):
    """Create the initial dynamic environment for Q-learning simulation."""
    read_graph(filename)

    for i, lot in enumerate(PARKING_LOTS):
        if lot in INVALID_LOTS:
            walk_km[i] = random.randint(W_MAX + 10, W_MAX + 20)
        else:
            walk_km[i] = random.randint(1, W_MAX)

    change_environment(force=True)


def change_environment(force=False):
    """Create or update the dynamic environment."""
    if not force and random.random() >= ENV_CHANGE_PROB:
        return

    for i in range(len(PARKING_LOTS)):
        p_avail[i] = random.choice([0, 1])
        price[i] = random.randint(10, 15)

    randomize_traffic()


def snapshot_environment():
    """Snapshot the current environment state."""
    return {
        'p_avail': p_avail.copy(),
        'price': price.copy(),
        'traffic': traffic.copy()
    }


def restore_environment(snapshot):
    """Restore the environment state from a snapshot."""
    global p_avail, price, traffic
    p_avail = snapshot['p_avail'].copy()
    price = snapshot['price'].copy()
    traffic = snapshot['traffic'].copy()


# ---------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------
def Dijkstra(start):
    """Compute shortest path distances and predecessors using Dijkstra's algorithm."""
    dist = {i: float('inf') for i in graph}
    prev = {i: None for i in graph}
    dist[start] = 0

    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        for v, w in graph[u]:
            level = traffic.get((u, v), "low")
            factor = traffic_multiplier(level)
            new_w = w * factor

            if dist[v] > d + new_w:
                dist[v] = d + new_w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    return dist, prev


def find_the_best_lot_by_cost(current_node, dist=None):
    """Find the best valid lot by full objective cost.

    Cost = travel distance + weighted walk distance + weighted price.
    """
    if dist is None:
        dist, _ = Dijkstra(current_node)

    best_lot = None
    best_cost = float('inf')

    for lot in PARKING_LOTS:
        idx = LOT_INDEX[lot]

        if p_avail[idx] == 0 or walk_km[idx] > W_MAX:
            continue

        if dist.get(lot, float('inf')) == float('inf'):
            continue

        cost = dist[lot] + LAMBDA_WALK * walk_km[idx] + LAMBDA_PRICE * price[idx]
        if cost < best_cost or (cost == best_cost and (best_lot is None or lot < best_lot)):
            best_cost = cost
            best_lot = lot

    return best_lot


def find_the_best_lot_by_distance(current_node, dist=None):
    """Find the nearest valid lot by shortest path distance only."""
    if dist is None:
        dist, _ = Dijkstra(current_node)

    best_lot = None
    best_dist = float('inf')

    for lot in PARKING_LOTS:
        idx = LOT_INDEX[lot]

        if p_avail[idx] == 0 or walk_km[idx] > W_MAX:
            continue

        lot_dist = dist.get(lot, float('inf'))
        if lot_dist == float('inf'):
            continue

        if lot_dist < best_dist or (lot_dist == best_dist and (best_lot is None or lot < best_lot)):
            best_dist = lot_dist
            best_lot = lot

    return best_lot


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------
def get_next_node(current, target, prev):
    """Get the next node on the shortest path from current to target."""
    if current == target:
        return None

    if prev[target] is None:
        return None

    path = []
    node = target
    while node != current and node is not None:
        path.append(node)
        node = prev[node]

    if node is None:
        return None

    return path[-1]


# ---------------------------------------------------------
# Reward for final state
# ---------------------------------------------------------
def is_current_lot_full(chosen_lot):
    if chosen_lot not in LOT_INDEX:
        return True
    idx = LOT_INDEX[chosen_lot]
    return p_avail[idx] == 0 or walk_km[idx] > W_MAX


def get_travel_time_bin(current_node, chosen_lot, dist):
    """Travel time to destination (chosen lot) using precomputed dist."""
    travel_time = dist.get(chosen_lot, float('inf'))

    if travel_time <= 20:
        return 0
    elif travel_time <= 50:
        return 1
    else:
        return 2


def get_availability_level():
    """Compute availability level for parking lots."""
    valid = 0
    available = 0

    for i in range(len(PARKING_LOTS)):
        if walk_km[i] <= W_MAX:
            valid += 1
            if p_avail[i] == 1:
                available += 1

    if valid == 0:
        return 0

    ratio = available / valid

    if ratio <= 0.3:
        return 0
    elif ratio <= 0.75:
        return 1
    else:
        return 2


def get_traffic_level(current_node, chosen_lot, dist, prev):
    """Average traffic along shortest path → discretized."""
    if dist.get(chosen_lot, float('inf')) == float('inf'):
        return 3

    path = []
    node = chosen_lot
    while node != current_node and node is not None:
        path.append(node)
        node = prev[node]

    if node is None:
        return 3

    path.append(current_node)
    path.reverse()

    total = 0
    count = 0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        level = traffic.get((u, v), "low")
        total += traffic_multiplier(level)
        count += 1

    if count == 0:
        return 1

    avg = total / count

    if avg <= 2:
        return 1
    elif avg <= 5:
        return 2
    else:
        return 3


def create_state(current_node, chosen_lot, dist, prev):
    """Build the agent state representation."""
    return (
        current_node,
        chosen_lot,
        is_current_lot_full(chosen_lot),
        get_travel_time_bin(current_node, chosen_lot, dist),
        get_availability_level(),
        get_traffic_level(current_node, chosen_lot, dist, prev)
    )


def reward(state, travel_time):
    current_node, chosen_lot, is_lot_full, *_ = state

    if current_node != chosen_lot:
        return -PENALTY - 1

    if current_node not in LOT_INDEX:
        return -PENALTY - 2

    idx = LOT_INDEX[current_node]

    if is_lot_full:
        return -PENALTY - 3

    return 500 - travel_time - LAMBDA_WALK * walk_km[idx] - LAMBDA_PRICE * price[idx]

def reward1(current_node, chosen_lot, travel_time):
    is_lot_full = is_current_lot_full(chosen_lot)

    if current_node != chosen_lot:
        return -PENALTY - 1

    if current_node not in LOT_INDEX:
        return -PENALTY - 2

    idx = LOT_INDEX[current_node]

    if is_lot_full:
        return -PENALTY - 3

    return 500 - travel_time - LAMBDA_WALK * walk_km[idx] - LAMBDA_PRICE * price[idx]