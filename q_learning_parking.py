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
EPISODES = 1000000
MAX_STEPS = 100
PERIOD = 20000
SAMPLE = 10000
TRY = 3

ALPHA = 0.1
ALPHA_MIN = 0.01
ALPHA_DECAY = 1.0   # no decay

EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_PERIOD = 200000

EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1 / EPSILON_DECAY_PERIOD)
GAMMA = 0.9

# Controlled dynamic environment, change by 50% chance
ENV_CHANGE_PROB = 0.5

script_dir = os.path.dirname(__file__)
OUTPUT_FILE = "output.txt"
OUTPUT = os.path.join(script_dir, OUTPUT_FILE)

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
LOTS = {"0": 21, "1": 22, "2": 23, "3": 24, "4": 25}
LOT_INDEX = {v: i for i, v in enumerate(LOTS.values())}

p_avail = [1] * 5
walk_km = [0] * 5
price = [0] * 5

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
    """
    read_graph()

    for i in range(5):
        if i < 3:
            walk_km[i] = random.randint(1, W_MAX)
        else:
            walk_km[i] = random.randint(W_MAX + 10, W_MAX + 20)

    change_environment(force=True)

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
# Trial
# ---------------------------------------------------------
def run_trial():
    current = 0
    chosen_lot = random.choice(list(LOTS.values()))
    travel_time = 0
    steps = 0

    change_environment(force=True)

    while steps < MAX_STEPS:
        steps += 1

        dist, prev = Dijkstra(current)
        state = create_state(current, chosen_lot, dist, prev)

        if state not in Q :
            return -1
        
        action = max(Q[state], key=Q[state].get)

        if action == "move":
            if current == chosen_lot or dist[chosen_lot] == float('inf'):
                return -1

            next_node = get_next_node(current, chosen_lot, prev)
            if next_node is None:
                return -1

            edge_weight = next((w for v, w in graph[current] if v == next_node), None)
            factor = traffic_multiplier(traffic.get((current, next_node), "low"))

            travel_time += edge_weight * factor
            current = next_node

        elif action == "switch":
            chosen_lot = random.choice(list(LOTS.values()))

        else:
            return max(reward(state, travel_time), -1)

        change_environment()

    return -1

# ---------------------------------------------------------
# Q-learning
# ---------------------------------------------------------
def q_learning_simulate():
    """
    Q-learning simulation

    * State:
        (
            current_node,
            chosen_lot,
            is_current_lot_full,
            travel_time_bin,
            availability_level,
            traffic_level
        )
    """
    global EPSILON, ALPHA

    actions = ["move", "switch", "park"]

    for ep in range(EPISODES):
        print(f"Trained {ep + 1}", end='\r')

        current = 0
        chosen_lot = random.choice(list(LOTS.values()))
        travel_time = 0

        change_environment(force=True)

        for _ in range(MAX_STEPS):
            dist, prev = Dijkstra(current)
            state = create_state(current, chosen_lot, dist, prev)
            r = 0
            done = False

            if state not in Q:
                Q[state] = {a: 0 for a in actions}

            if random.random() < EPSILON:
                action = random.choice(actions)
            else:
                action = max(Q[state], key=Q[state].get)

            if action == "move":
                if current == chosen_lot or dist[chosen_lot] == float('inf'):
                    r = -50
                    done = True
                else:
                    next_node = get_next_node(current, chosen_lot, prev)
                    edge_weight = next((w for v, w in graph[current] if v == next_node), None)

                    factor = traffic_multiplier(traffic.get((current, next_node), "low"))
                    step_time = edge_weight * factor

                    travel_time += step_time
                    current = next_node

                    #Recompute after changing node
                    dist, prev = Dijkstra(current)

                    r = -step_time * 0.1
                    done = False

            elif action == "switch":
                chosen_lot = random.choice(list(LOTS.values()))
                r = -5
                done = False

            else:
                r = reward(state, travel_time)
                done = True

            next_state = create_state(current, chosen_lot, dist, prev)

            if next_state not in Q:
                Q[next_state] = {a: 0 for a in actions}

            max_next = 0 if done else max(Q[next_state].values())

            Q[state][action] += ALPHA * (
                r + GAMMA * max_next - Q[state][action]
            )

            change_environment()

            if done:
                break

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        ALPHA = max(ALPHA_MIN, ALPHA * ALPHA_DECAY)
        
        if (((ep + 1 <= 20000) and (ep + 1) % 500 == 0) or (ep + 1) % PERIOD == 0) :
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
            if avg > 0:
                avg_return /= avg
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

    msg = "Trial#, Alpha, Epsilon, Trial 1, Avg 1, Trial 2, Avg 2, Trial 3, Avg 3, AverageSuccess, AverageReturn"
    write_result(msg)

    create_environment()
    q_learning_simulate()

if __name__ == "__main__":
    main()
