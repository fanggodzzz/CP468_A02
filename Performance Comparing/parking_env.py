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
LOTS = {"1": 21, "2": 22, "3": 23, "4": 24, "5": 25}

# reverse map for fast lookup
LOT_INDEX = {v: i for i, v in enumerate(LOTS.values())}

p_avail = [1, 1, 1, 1, 1]
walk_km = [0, 0, 0, 0, 0]
price = [0, 0, 0, 0, 0]

graph = {}
traffic = {}


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
    Randomize traffic levels for edges in the graph.

    Each edge has a 30% chance of having traffic ("low", "medium", "high").
    """
    global traffic
    traffic = {}

    for u in graph:
        for v, _ in graph[u]:
            if random.random() < 0.3:
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


def change_environment():
    """
    Update the dynamic environment.

    Randomizes parking lot availability (0/1) and prices (10-15),
    and randomizes traffic levels.
    """
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
# Reward
# ---------------------------------------------------------
def reward(state, travel_time):
    """
    Calculate the reward for reaching a parking lot state.

    Args:
        state (tuple): (node, chosen_lot, avail_tuple)
        travel_time (float): Total travel time to reach the lot.

    Returns:
        float: Reward value. Returns -PENALTY if not at chosen lot, lot unavailable, or walking distance too far.
    """
    node, chosen_lot, avail_tuple = state

    if node not in LOTS.values() or node != chosen_lot:
        return -PENALTY - 5

    idx = LOT_INDEX[node]

    if avail_tuple[idx] == 0 or walk_km[idx] > W_MAX:
        return -PENALTY - 6

    return 500 - travel_time - LAMBDA_WALK * walk_km[idx] - LAMBDA_PRICE * price[idx]


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
