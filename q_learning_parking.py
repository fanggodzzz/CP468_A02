import random

# =========================================================
# CP468 Assignment 2 - Task 2
# Q-Learning for Parking Lot Recommendation
# Group 20
# =========================================================

# ---------------------------------------------------------
# Sample test input
# ---------------------------------------------------------


LOTS = {"1": 21, "2": 22, "3": 23, "4": 24, "5": 25} # Dictionary mapping parking lots to nodes
p_avail = [1, 1, 1, 1, 1]  # parking availability for each parking lot

# Drive time to each parking lot at each time step t
drive_time = {
    "P1": [9, 10, 12, 11],
    "P2": [11, 10, 9, 9],
    "P3": [8, 8, 8, 8],
}

# Walking distance from parking lot to destination
walk_km = {
    "P1": 0.4,
    "P2": 0.2,
    "P3": 0.6,
}

# Parking price at each time step
price = {
    "P1": [8, 8, 9, 9],
    "P2": [12, 10, 9, 9],
    "P3": [5, 5, 5, 5],
}

# Availability at each time step (1 = available, 0 = full)
avail = {
    "P1": [1, 1, 0, 1],
    "P2": [0, 1, 1, 0],
    "P3": [1, 0, 1, 1],
}

# ---------------------------------------------------------
# Constraints and weights
# ---------------------------------------------------------
W_MAX = 10
LAMBDA_WALK = 10.0
LAMBDA_PRICE = 1.0
PENALTY = 10**6

# ---------------------------------------------------------
# Q-learning parameters
# ---------------------------------------------------------
ALPHA = 0.1      # learning rate
GAMMA = 0.9      # discount factor
EPSILON = 0.2    # exploration rate
EPISODES = 5000

# States are time steps: 0, 1, 2, 3
TIME_STEPS = len(drive_time["P1"])

def cost(lot, t):
    """
    Cost function based on Assignment 1:
    J(p,t) = DriveTime + lambda_walk * WalkDist + lambda_price * Price + Penalty
    """
    penalty = 0

    if walk_km[lot] > W_MAX:
        penalty += PENALTY

    if avail[lot][t] == 0:
        penalty += PENALTY

    return (
        drive_time[lot][t]
        + LAMBDA_WALK * walk_km[lot]
        + LAMBDA_PRICE * price[lot][t]
        + penalty
    )


def reward(lot, t):
    """
    Q-learning maximizes reward, so reward = -cost.
    Lower cost means higher reward.
    """
    return -cost(lot, t)


def is_terminal_state(t):
    """
    Last time step is terminal.
    """
    return t == TIME_STEPS - 1





def create_environment():
    """
    Create dynamic environment for Q-learning simulation

    * Network: 
        - Graph with 21 nodes + 5 parking nodes
            + Node 0:         starting node
            + Node 1 - 20:    intersections
            + Node 21 - 25:   parking lots
    * Parking lots:
        - Parking lot 1 - 3 (Node 21 - 23): Within W_distance
        - Parking lot 4 - 5 (Node 24 - 25): Out of W_MAX distance
        - Price for each lots - randomize from 10 - 15 (dynamically change)
        - Availability: 0 or 1 (dynamically change)   
    * Traffic (Undirected Edges):
        - Fixed weight edges (randomly choose from 5 - 10)
        - During simulation, traffic dynamicity: 
            + Low:     weight * 1
            + Medium:  weight * 2
            + High:    weight * 5
    """
    pass

def change_environment():
    """
    Create dynamic environment for Q-learning simulation

    * Network: 
        - Graph with 21 nodes + 5 parking nodes
            + Node 0: starting node
            + Node 1 - 20: intersections
            + Node 21 - 25: parking lots
    * Parking lots:
        - Parking lot 1 - 3 (Node 21 - 23): Within W_distance
        - Parking lot 4 - 5 (Node 24 - 25): Out of W_MAX distance
        - Price for each lots - randomize from 10 - 20 (dynamically change)
        - Availability: 0 or 1 (dynamically change)   
    * Traffic (Undirected Edges):
        - Fixed weight edges (randomly choose from 5 - 10)
        - During simulation, traffic dynamicity: 
            + Low   (0): weight * 1
            + Medium(1): weight * 2
            + High  (2): weight * 5
    """
    pass

def q_learning_simulate():
    """

    """
    pass

def print_result():
    pass

def main():
    create_environment()
    q_learning_simulate()
    print_result()
    pass

if __name__ == "__main__":
    main()
