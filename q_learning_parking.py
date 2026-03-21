import random

# =========================================================
# CP468 Assignment 2 - Task 2
# Q-Learning for Parking Lot Recommendation
# Group 20
# =========================================================

# ---------------------------------------------------------
# Sample test input
# ---------------------------------------------------------
LOTS = ["P1", "P2", "P3"]

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
W_MAX = 0.5
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



