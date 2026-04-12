"""
Training Module for Q-Learning Parking System

This module handles the training of the Q-learning agent for parking lot recommendation.
It includes Q-table initialization, training loop, and saving the trained Q-table.
"""

import argparse
import numpy as np
import random
import os

import parking_env as env

ACTIONS = ["move", "switch", "park"]

# ---------------------------------------------------------
# Q-learning parameters
# ---------------------------------------------------------
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_PERIOD = 100000
EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1 / EPSILON_DECAY_PERIOD)
ALPHA_MIN = 0.01    
ALPHA_DECAY = 1.0
EPISODES = 750000
MAX_STEPS = 100

script_dir = os.path.dirname(os.path.abspath(__file__))

def save_q_table(q_table, filename):
    """
    Save the Q-table to a numpy file.

    Args:
        q_table (dict): The Q-table to save.
        filename (str): Path to save the file.
    """
    filename = os.path.join(script_dir, filename)
    np.save(filename, q_table, allow_pickle=True)


def q_learning_train(episodes, max_steps, graph_file):
    """
    Train the Q-learning agent.

    Args:
        episodes (int): Number of training episodes.
        output_file (str): Path to save the trained Q-table.
        max_steps (int): Maximum steps per episode.
        graph_file (str): Path to the graph input file.
    """
    Q = {}
    epsilon = EPSILON
    alpha = ALPHA

    graph_file = os.path.join(script_dir, graph_file)
    env.create_environment(graph_file)

    for ep in range(episodes):
        msg = f"Trained {ep + 1} episodes"
        print(msg, end='\r')

        current = 0
        chosen_lot = random.choice(env.PARKING_LOTS)
        done = False
        steps = 0
        travel_time = 0

        env.change_environment()

        while not done and steps < max_steps:
            steps += 1

            dist, prev = env.Dijkstra(current)
            state = env.create_state(current, chosen_lot, dist, prev)

            if state not in Q:
                Q[state] = {a: 0 for a in ACTIONS}

            # e-greedy
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = max(Q[state], key=Q[state].get)

            if action == "move":
                if current == chosen_lot or dist[chosen_lot] == float('inf'):
                    r = -50
                    done = True
                else:
                    next_node = env.get_next_node(current, chosen_lot, prev)
                    edge_weight = next((w for v, w in env.graph[current] if v == next_node), None)

                    factor = env.traffic_multiplier(env.traffic.get((current, next_node), "low"))
                    step_time = edge_weight * factor

                    travel_time += step_time
                    current = next_node

                    #Recompute after changing node
                    dist, prev = env.Dijkstra(current)

                    r = -step_time * 0.1
                    done = False

            elif action == "switch":
                chosen_lot = env.find_the_best_lot_by_distance(current, dist)
                if chosen_lot is None:
                    chosen_lot = random.choice(env.PARKING_LOTS)
                r = -5
                done = False

            else:
                r = env.reward(state, travel_time)
                done = True

            next_state = env.create_state(current, chosen_lot, dist, prev)

            if next_state not in Q:
                Q[next_state] = {a: 0 for a in ACTIONS}

            max_next = 0 if done else max(Q[next_state].values())
            Q[state][action] += alpha * (r + GAMMA * max_next - Q[state][action])
            current = next_state[0]
            env.change_environment()

        # Reduce epsilon per episode
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        # Reduce alpha per episode
        if alpha > ALPHA_MIN:
            alpha *= ALPHA_DECAY

    save_q_table(Q, "q_table.npy")
    print(f"Saved Q-table to q_table.npy")


def main():
    """
    Main function to parse arguments and start training.
    """
    parser = argparse.ArgumentParser(description="Train Q-learning and save the Q table.")
    parser.add_argument("graph", help="Path to the graph input file")
    parser.add_argument("episodes", type=int, help="Number of training episodes")
    args = parser.parse_args()

    print("Starting Q-learning training...")

    q_learning_train(args.episodes, MAX_STEPS, args.graph)

    print("-----Training done-----")


if __name__ == "__main__":
    main()
