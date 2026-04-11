"""
Parking Environment Module

This module provides the environment setup, graph reading, traffic simulation,
and utility functions for the Q-learning parking lot recommendation system.
"""

import copy
import heapq
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
W_MAX = 10
LAMBDA_WALK = 10.0
LAMBDA_PRICE = 1.0
PENALTY = 5000  # Important

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
LOTS = {"0": 21, "1": 22, "2": 23, "3": 24, "4": 25}

# reverse map for fast lookup
LOT_INDEX = {v: i for i, v in enumerate(LOTS.values())}

p_avail = [1] * 5
walk_km = [0, 0, 0, 0, 0]
price = [0, 0, 0, 0, 0]

graph = {}
traffic = {}

# Controlled dynamic environment, change by 50% chance
ENV_CHANGE_PROB = 0.5


# ---------------------------------------------------------
# Read graph
# In this code, the graph is undirected, but can still
# work with directed graph
# ---------------------------------------------------------
def read_graph(filename="graph.txt"):
    """
    Read the graph from a file.

    The graph is undirected and stored as an adjacency list.

    Args:
        filename (str): Path to the graph file. Default is "graph.txt".
    """
    global graph
    graph = {}

    with open(filename, "r") as f:
        n, m = map(int, f.readline().split())

        for i in range(n):
            graph[i] = []

        for _ in range(m):
            u, v, w = map(int, f.readline().split())
            graph[u].append((v, w))
            graph[v].append((u, w))


# ---------------------------------------------------------
# Traffic
# ---------------------------------------------------------
def traffic_multiplier(level):
    """
    Get the traffic multiplier based on traffic level.

    Args:
        level (str): Traffic level ("low", "medium", "high").

    Returns:
        int: Multiplier for edge weight (1, 3, or 7).
    """
    if level == "low":
        return 1
    elif level == "medium":
        return 3
    else:
        return 7


def randomize_traffic():
    """
    Randomize the traffic for edges in the graph.
    """
    global traffic
    traffic = {}

    for u in graph:
        for v, _ in graph[u]:
            traffic[(u, v)] = random.choice(["low", "medium", "high"])


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
def create_environment():
    """
    Create the initial dynamic environment for Q-learning simulation.

    Reads the graph, sets random walking distances for parking lots,
    and calls change_environment to initialize parking availability and prices.
    """
    read_graph()

    for i in range(5):
        if i < 3:
            walk_km[i] = random.randint(1, W_MAX)
        else:
            walk_km[i] = random.randint(W_MAX + 10, W_MAX + 20)

    change_environment()


def change_environment(force=False):
    """
    Create or update the dynamic environment.
    """
    if not force and random.random() >= ENV_CHANGE_PROB:
        return

    global p_avail, price

    for i in range(5):
        p_avail[i] = random.choice([0, 1])
        price[i] = random.randint(10, 15)

    randomize_traffic()


# ---------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------
def Dijkstra(start):
    """
    Compute shortest path distances and predecessors using Dijkstra's algorithm,
    considering dynamic traffic levels.

    Args:
        start (int): Starting node.

    Returns:
        tuple: (dist, prev) where dist is distance dict, prev is predecessor dict.
    """
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


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------
def get_next_node(current, target, prev):
    """
    Get the next node on the shortest path from current to target.

    Args:
        current (int): Current node.
        target (int): Target node.
        prev (dict): Predecessor dictionary from Dijkstra.

    Returns:
        int or None: Next node to move to, or None if no path.
    """
    if prev[target] is None:
        return None

    path = []
    while target != current:
        path.append(target)
        target = prev[target]

    return path[-1]


# ---------------------------------------------------------
# Reward for final state
# ---------------------------------------------------------
def reward(state, travel_time):
    current_node, chosen_lot, is_lot_full, *_ = state

    if current_node != chosen_lot:
        return -PENALTY
    
    if current_node not in LOT_INDEX:
        return -PENALTY

    idx = LOT_INDEX[current_node]

    if is_lot_full:
        return -PENALTY

    return 500 - travel_time - LAMBDA_WALK * walk_km[idx] - LAMBDA_PRICE * price[idx]


# ---------------------------------------------------------
# State
# ---------------------------------------------------------
def is_current_lot_full(chosen_lot):
    if chosen_lot not in LOTS.values():
        return True
    idx = LOT_INDEX[chosen_lot]
    return p_avail[idx] == 0 or walk_km[idx] > W_MAX


def get_travel_time_bin(current_node, chosen_lot, dist):
    """
    Travel time to destination (chosen lot) using precomputed dist.

    Bins:
        0: near (<= 20)
        1: medium (20 - 50)
        2: far (50+)
    """
    travel_time = dist.get(chosen_lot, float('inf'))

    if travel_time <= 20:
        return 0
    elif travel_time <= 50:
        return 1
    else:
        return 2

def get_availability_level():
    """
    Availability = (# available & valid lots) / (# valid lots)

    Valid lot = walk_km <= W_MAX

    Levels:
        0: low    (<= 30%)
        1: medium (30% - 75%)
        2: high   (> 75%)
    """
    valid = 0
    available = 0

    for i in range(5):
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
    """
    Average traffic along shortest path → discretized

    Output:
        1: low
        2: medium
        3: heavy
    """
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
    """
    State:
        (
            current_node,
            chosen_lot,
            is_current_lot_full,
            travel_time_bin,
            availability_level,
            traffic_level
        )
    """
    return (
        current_node,
        chosen_lot,
        is_current_lot_full(chosen_lot),
        get_travel_time_bin(current_node, chosen_lot, dist),
        get_availability_level(),
        get_traffic_level(current_node, chosen_lot, dist, prev)
    )


def snapshot_environment():
    """
    Take a snapshot of the current environment state.

    Returns:
        dict: Snapshot containing traffic, p_avail, price, walk_km.
    """
    return {
        "traffic": copy.deepcopy(traffic),
        "p_avail": list(p_avail),
        "price": list(price),
        "walk_km": list(walk_km),
    }


def restore_environment(snapshot):
    """
    Restore the environment to a previous snapshot.

    Args:
        snapshot (dict): Snapshot from snapshot_environment.
    """
    traffic.clear()
    traffic.update(copy.deepcopy(snapshot["traffic"]))
    p_avail[:] = list(snapshot["p_avail"])
    price[:] = list(snapshot["price"])
    walk_km[:] = list(snapshot["walk_km"])
