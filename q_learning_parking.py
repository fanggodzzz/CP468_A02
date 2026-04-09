import argparse
import random
import heapq
import sys
import os

# ---------------------------------------------------------
# Constraints and weights
# ---------------------------------------------------------
W_MAX = 10
LAMBDA_WALK = 10.0
LAMBDA_PRICE = 1.0
PENALTY = 5000

# ---------------------------------------------------------
# Q-learning parameters 
# ---------------------------------------------------------
EPISODES = 500000
MAX_STEPS = 100
PERIOD = 5000
SAMPLE = 5000
TRY = 3

ALPHA = 0.1
ALPHA_MIN = 0.01
ALPHA_DECAY = 1.0   # no decay

EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_PERIOD = 100000

EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1 / EPSILON_DECAY_PERIOD)
GAMMA = 0.9

# Controlled dynamic environment, change by 25% chance
ENV_CHANGE_PROB = 0.25

script_dir = os.path.dirname(__file__)
OUTPUT_FILE = "output.txt"
OUTPUT = os.path.join(script_dir, OUTPUT_FILE)

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
LOTS = {"1": 21, "2": 22, "3": 23, "4": 24, "5": 25}
LOT_INDEX = {v: i for i, v in enumerate(LOTS.values())}

p_avail = [1] * 6
walk_km = [0] * 6
price = [0] * 6

graph = {}
traffic = {}
Q = {}

# ---------------------------------------------------------
# File utils
# ---------------------------------------------------------
def clear_file():
    with open(OUTPUT, "w") as f:
        f.write("")

def write_result(log):
    with open(OUTPUT, "a") as f:
        f.write(f"{log}\n")

# ---------------------------------------------------------
# Graph
# ---------------------------------------------------------
def read_graph(filename="graph.txt"):
    """
    Read graph from file.
    
    The graph is undirected and stored as an adjacency list.
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
    Get traffic multiplier based on level.
    
    Args:
        level (str): Traffic level ("low", "medium", "high").
    
    Returns:
        int: Multiplier for edge weight (1, 3, or 7).
    """
    return {"low": 1, "medium": 3, "high": 7}[level]

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
    Create dynamic environment for Q-learning simulation

    * Network: 
        - Graph with 21 nodes + 5 parking nodes
            + Node 0:         starting node
            + Node 1 - 20:    intersections
            + Node 21 - 25:   parking lots
    * Parking lots:
        - Parking lot 1 - 3 (Node 21 - 23): Random within W_MAX distance
        - Parking lot 4 - 5 (Node 24 - 25): Random out of W_MAX distance  
    * Traffic (Undirected Edges):
        - Read edges from "graph.txt"

    Then call change_environment()
    """
    read_graph()

    for i in range(1, 6):
        if i <= 3:
            walk_km[i] = random.randint(1, W_MAX)
        else:
            walk_km[i] = random.randint(W_MAX + 10, W_MAX + 20)

    change_environment(force=True)

def change_environment(force=False):
    """
    Create or update the dynamic environment.

    If force is False, the environment only changes with probability ENV_CHANGE_PROB.
    If force is True, the environment changes unconditionally.
    """
    if not force and random.random() >= ENV_CHANGE_PROB:
        return

    global p_avail, price

    for i in range(1, 6):
        p_avail[i] = random.choice([0, 1])
        price[i] = random.randint(10, 15)

    randomize_traffic()

# ---------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------
def Dijkstra(start):
    """
    Compute shortest path with dynamic traffic
    """
    dist = {i: float('inf') for i in graph}
    prev = {i: None for i in graph}
    dist[start] = 0

    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)

        for v, w in graph[u]:
            level = traffic.get((u, v), "low")
            new_w = w * traffic_multiplier(level)

            if dist[v] > d + new_w:
                dist[v] = d + new_w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    return dist, prev

def get_next_node(current, target, prev):
    """
    Get next step toward target using shortest path
    """
    if prev[target] is None:
        return None

    path = []
    while target != current:
        path.append(target)
        target = prev[target]

    return path[-1]

# ---------------------------------------------------------
# State
# ---------------------------------------------------------
def is_current_lot_full(chosen_lot):
    if chosen_lot not in LOTS.values():
        return True
    idx = LOT_INDEX[chosen_lot]
    return p_avail[idx] == 0 or walk_km[idx] > W_MAX

def is_any_lot_available_nearby(dist, chosen_lot):
    chosen_dist = dist.get(chosen_lot, float('inf'))
    if chosen_dist == float('inf'):
        return False

    threshold = chosen_dist // 2 * 3  # Nearby means not farther than 1.5 times the current distance
    for lot_node in LOTS.values():
        idx = LOT_INDEX[lot_node]
        if p_avail[idx] == 1 and walk_km[idx] <= W_MAX:
            lot_dist = dist.get(lot_node, float('inf'))
            if lot_dist != float('inf') and lot_dist <= threshold:
                return True
    return False

def create_state(current_node, chosen_lot, dist):
    return (
        current_node,
        chosen_lot,
        is_current_lot_full(chosen_lot),
        is_any_lot_available_nearby(dist, chosen_lot)
    )

# ---------------------------------------------------------
# Reward at final state
# ---------------------------------------------------------
def reward(state, travel_time):
    current_node, chosen_lot, is_lot_full, _ = state

    if current_node != chosen_lot:
        return -PENALTY

    idx = LOT_INDEX[current_node]

    if is_lot_full:
        return -PENALTY

    return 500 - travel_time - LAMBDA_WALK * walk_km[idx] - LAMBDA_PRICE * price[idx]

# ---------------------------------------------------------
# Trial
# ---------------------------------------------------------
def run_trial():
    current = 0
    chosen_lot = random.choice(list(LOTS.values()))
    travel_time = 0
    steps = 0

    change_environment()

    while steps < MAX_STEPS:
        steps += 1

        dist, prev = Dijkstra(current)
        state = create_state(current, chosen_lot, dist)

        if state not in Q or random.random() < 0.1:
            action = random.choice(["move", "switch", "park"])
        else:
            action = max(Q[state], key=Q[state].get)

        if action == "move":
            if dist[chosen_lot] == float('inf'):
                return -1

            next_node = get_next_node(current, chosen_lot, prev)
            if next_node is None:
                return -1

            edge_weight = next((w for v, w in graph[current] if v == next_node), None)
            if edge_weight is None:
                return -1

            factor = traffic_multiplier(traffic.get((current, next_node), "low"))
            step_time = edge_weight * factor

            travel_time += step_time
            current = next_node

        elif action == "switch":
            chosen_lot = random.choice(list(LOTS.values()))

        else:
            return max(reward(state, travel_time), -1)

        if random.random() < ENV_CHANGE_PROB:
            change_environment()

    return -1

# ---------------------------------------------------------
# Q-learning
# ---------------------------------------------------------
def q_learning_simulate():
    """
    Q-learning simulation

    * State:
        (current_node, chosen_lot, is_current_lot_full, is_any_lot_available_nearby)
    * Action: 
        - Move to next node (Use Dijkstra to determine the path and move to the next node)
        - Change parking lot
        - Parking - Done 
    """
    global EPSILON, ALPHA

    actions = ["move", "switch", "park"]

    for ep in range(EPISODES):
        print(f"Trained {ep}", end='\r')

        current = 0
        chosen_lot = random.choice(list(LOTS.values()))
        travel_time = 0

        change_environment()

        for _ in range(MAX_STEPS):
            dist, prev = Dijkstra(current)
            state = create_state(current, chosen_lot, dist)

            if state not in Q:
                Q[state] = {a: 0 for a in actions}

            if random.random() < EPSILON:
                action = random.choice(actions)
            else:
                action = max(Q[state], key=Q[state].get)

            # Action: Move
            if action == "move":
                if current == chosen_lot or dist[chosen_lot] == float('inf'):
                    r = -100
                    done = True
                else:
                    next_node = get_next_node(current, chosen_lot, prev)

                    edge_weight = next((w for v, w in graph[current] if v == next_node), None)

                    if edge_weight is None:
                        r = -100
                        done = True
                    else:
                        factor = traffic_multiplier(traffic.get((current, next_node), "low"))
                        step_time = edge_weight * factor

                        travel_time += step_time
                        current = next_node

                        r = -step_time * 0.1
                        done = False

            # Action: Switch
            elif action == "switch":
                chosen_lot = random.choice(list(LOTS.values()))
                r = -5
                done = False

            # Action: Park
            else:
                r = reward(state, travel_time)
                done = True

            next_state = create_state(current, chosen_lot, dist)

            if next_state not in Q:
                Q[next_state] = {a: 0 for a in actions}

            max_next = 0 if done else max(Q[next_state].values())

            Q[state][action] += ALPHA * (
                r + GAMMA * max_next - Q[state][action]
            )

            if random.random() < ENV_CHANGE_PROB:
                change_environment()

            if done:
                break

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        ALPHA = max(ALPHA_MIN, ALPHA * ALPHA_DECAY)

        if ((ep + 1) % PERIOD == 0) :
            avg = 0
            avg_return = 0
            msg_ep = f"{ep + 1}, {ALPHA:.4f}, {EPSILON:.4f}"
            for _ in range(TRY):
                count = 0
                avg_return1 = 0
                for _ in range(SAMPLE):
                    trial = run_trial()
                    if (trial >=  0):
                        count += 1
                        avg_return1 += trial
                avg += count
                msg_ep += f", {count}, {avg_return1 / count if count > 0 else 0:.2f}"
                avg_return += avg_return1
            avg_return //= avg
            avg //= TRY
            msg_ep += f", {avg}, {avg_return:.2f}"
            write_result(msg_ep)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    """
    Main function to run the Q-learning training.
    """

    clear_file()

    msg = "Alpha, Epsilon, Trial 1, Avg 1, Trial 2, Avg 2, Trial 3, Avg 3, AverageSuccess, AverageReturn"
    write_result(msg)

    create_environment()
    q_learning_simulate()

if __name__ == "__main__":
    main()
