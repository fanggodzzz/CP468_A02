import random
import heapq
import sys

# ---------------------------------------------------------
# Constraints and weights
# ---------------------------------------------------------
W_MAX = 10
LAMBDA_WALK = 10.0
LAMBDA_PRICE = 1.0
PENALTY = 5000  #Important 

# ---------------------------------------------------------
# Q-learning parameters
# ---------------------------------------------------------
ALPHA = 0.5
GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99998  
ALPHA_MIN = 0.05
ALPHA_DECAY = 0.99998    
EPISODES = 10000
MAX_STEPS = 100
TRY = 7500
PERIOD = 10000
SAMPLE = 2
OUTPUT = "output.txt"
DECAY_PERIOD = -1

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
Q = {}

travel_time = 0

# ---------------------------------------------------------
# Clear output file
# ---------------------------------------------------------
def clear_file():
    with open(OUTPUT, "w") as f:
        f.write("")

# ---------------------------------------------------------
# Write result to OUTPUT file
# ---------------------------------------------------------
def write_result(log):
    with open(OUTPUT, "a") as f:
        f.write(f"{log}\n")

# ---------------------------------------------------------
# Read graph
# In this code, the graph is undirected, but can still 
# work with directed graph
# ---------------------------------------------------------

def read_graph(filename="graph.txt"):
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
    if level == "low":
        return 1
    elif level == "medium":
        return 3
    else:
        return 7

def randomize_traffic():
    """
    Randomize the traffic
    """

    global traffic
    traffic = {}

    for u in graph:
        for v, w in graph[u]:
            if random.random() < 0.3:
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

    for i in range(5):
        if i < 3:
            walk_km[i] = random.randint(1, W_MAX)
        else:
            walk_km[i] = random.randint(W_MAX + 10, W_MAX + 20)

    change_environment()

def change_environment():
    """
    Create the dynamicity for the environment

    * Parking lot dynamicity:
        - Random the availability (0 / 1)
        - Random the price (10 - 15)

    * Trafic dynamicity
        - Change traffic for each edge:
            + Low:     0 (weight * 1)
            + Medium:  1 (weight * 3)
            + High:    2 (weight * 7)
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
# Reward
# ---------------------------------------------------------
def reward(state, travel_time):
    """
    Immediate reward for each state

    If node is not parking lot --> return 0
    Else at parking lot:
        - plus 500 if the the lot is available (avail = 1)
        - minus 10^6 if the the lot is not available (avail = 0) 
            or parking lot is out of max distance (walk_km > W_MAX)
        - minus time travels from node 0 to the parking lot
        - minus the price of the parking lot 
        - minus the walking distance to destination D
    """

    node, chosen_lot, avail_tuple = state

    if node not in LOTS.values():
        return -PENALTY

    idx = LOT_INDEX[node]

    if avail_tuple[idx] == 0 or walk_km[idx] > W_MAX:
        return -PENALTY

    return 500 - travel_time - LAMBDA_WALK * walk_km[idx] - LAMBDA_PRICE * price[idx]

# ---------------------------------------------------------
# Trial - Almost same code as Q-learning
# ---------------------------------------------------------

def run_trial():

    # print("\n--- Trial Run ---")

    current = 0
    chosen_lot = random.choice(list(LOTS.values()))
    travel_time = 0
    steps = 0

    done = False

    change_environment()

    while not done and steps < MAX_STEPS:

        steps += 1

        dist, prev = Dijkstra(current)

        state = (current, chosen_lot, tuple(p_avail))

        if state not in Q:
            action = random.choice(["move", "switch", "park"])
        else:
            action = max(Q[state], key=Q[state].get)

        # print(f"\nStep {steps}")
        # print("Current node:", current)
        # print("Chosen lot:", chosen_lot)
        # print("Action:", action)

        if action == "move":

            if dist[chosen_lot] == float('inf'):
                # print("No route to lot.")
                continue

            next_node = get_next_node(current, chosen_lot, prev)

            if next_node is None:
                # print("Path reconstruction failed.")
                continue

            edge_weight = None
            for v, w in graph[current]:
                if v == next_node:
                    edge_weight = w
                    break

            if edge_weight is None:
                # print("Invalid edge.")
                # write_result("Invalid edge.")
                continue

            level = traffic.get((current, next_node), "low")
            factor = traffic_multiplier(level)

            step_time = edge_weight * factor
            travel_time += step_time

            # print("Moving to:", next_node, "| travel time:", step_time)

            current = next_node

        elif action == "switch":

            chosen_lot = random.choice(list(LOTS.values()))
            # print("Switching lot →", chosen_lot)

        else:

            return reward(state, travel_time)

        change_environment()

    if not done:
        # print("\nTrial failed (max steps reached)")
        return -1
    

# ---------------------------------------------------------
# Q-learning
# ---------------------------------------------------------
def q_learning_simulate():
    """
    Q-learning simulation

    * State:
        current node
        chosen parking lot
        Availability of each parking lot
    * Action: 
        - Move to next node (Use Dijkstra to determine the path and move to the next node)
        - Change parking lot
        - Parking - Done 
    """
    global EPSILON, travel_time, ALPHA

    actions = ["move", "switch", "park"]

    for ep in range(EPISODES):
        msg = f"Trained {ep} episodes"
        print(msg, end='\r')
        current = 0
        chosen_lot = random.choice(list(LOTS.values()))
        done = False
        steps = 0

        travel_time = 0

        # Initialize new environment for new episode
        change_environment() 

        while not done and steps < MAX_STEPS:

            steps += 1

            dist, prev = Dijkstra(current)

            state = (current, chosen_lot, tuple(p_avail))

            if state not in Q:
                Q[state] = {a: 0 for a in actions}

            # e-greedy
            if random.random() < EPSILON:
                action = random.choice(actions)
            else:
                action = max(Q[state], key=Q[state].get)

            # Action: Move
            if action == "move":
                # Prevent moving if at the parking lot
                if current == chosen_lot:  
                    r = -5
                    next_state = (current, chosen_lot, tuple(p_avail))
                    done = False

                # Prevent further running if not reaching the lot
                elif dist[chosen_lot] == float('inf'):  
                    r = -100
                    next_state = (current, chosen_lot, tuple(p_avail))
                    done = False

                else:
                    # Get the next node on the shortest path
                    next_node = get_next_node(current, chosen_lot, prev) 

                    # Get the weight and calculate the time travel for that edge
                    edge_weight = None 
                    for v, w in graph[current]:
                        if v == next_node:
                            edge_weight = w
                            break

                    level = traffic.get((current, next_node), "low")
                    factor = traffic_multiplier(level)
                    step_time = edge_weight * factor

                    # Calculate the cumulative time travel
                    travel_time += step_time 

                    # Transition
                    next_state = (next_node, chosen_lot, tuple(p_avail))
                    r = -step_time
                    done = False

            # Action: Switch
            elif action == "switch":

                chosen_lot = random.choice(list(LOTS.values()))

                next_state = (current, chosen_lot, tuple(p_avail))

                # Small penalty for switching
                r = -5
                  
                done = False

            # Action: Park
            else:

                next_state = (current, chosen_lot, tuple(p_avail))

                # Prevent parking at anywhere    
                if current != chosen_lot:
                    r = -100
                    done = False
                
                #Calculate reward for final state
                else:
                    r = reward(next_state, travel_time)
                    done = True

            if next_state not in Q:
                Q[next_state] = {a: 0 for a in actions}

            max_next = 0 if done else max(Q[next_state].values())

            Q[state][action] += ALPHA * (
                r + GAMMA * max_next - Q[state][action]
            )

            current = next_state[0]

            # Make the environment dynamically
            change_environment()

        # Reduce epsilon per episode
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        # Reduce alpha per episode 
        if ALPHA > ALPHA_MIN:
            ALPHA*= ALPHA_DECAY

        if ((ep + 1) % PERIOD == 0) :
            print()
            write_result(f"Trained {ep + 1} episodes")
            msg_ep = f"Epsilon: {EPSILON:.4f}, Alpha: {ALPHA:.4f}"
            print(msg_ep)
            write_result(msg_ep)
            for x in range(SAMPLE):
                count = 0
                for i in range(TRY):
                    msg_try = f"Tried {i}"
                    print(msg_try, end='\r')
                    if (i + 1 == TRY): 
                        write_result(msg_try)
                    count += (run_trial() > -1)
                print()
                print(count)
                write_result(str(count))

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    clear_file()

    global ALPHA, EPSILON, ALPHA_DECAY, EPSILON_DECAY, DECAY_PERIOD
    alpha_msg = f"ALPHA: {ALPHA}, ALPHA_DECAY: {ALPHA_DECAY}, EPSILON: {EPSILON}, EPSILON_DECAY: {EPSILON_DECAY}"
    print(alpha_msg)
    write_result(alpha_msg)

    create_environment()

    q_learning_simulate()

    training_done_msg = "Training done"
    print("\n" + training_done_msg)
    write_result(training_done_msg)

    print(run_trial())

if __name__ == "__main__": 
    main()