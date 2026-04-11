import argparse
import numpy as np
import random

import parking_env as env

ACTIONS = ["move", "switch", "park"]
MAX_STEPS = 100


def load_q_table(filename):
    data = np.load(filename, allow_pickle=True)
    if isinstance(data, np.ndarray):
        return data.item()
    return data


def run_trial_with_q(q_table, max_steps):
    current = 0
    chosen_lot = random.choice(list(env.LOTS.values()))
    travel_time = 0
    steps = 0
    done = False

    env.change_environment()

    while not done and steps < max_steps:
        steps += 1

        dist, prev = env.Dijkstra(current)
        state = env.create_state(current, chosen_lot, dist, prev)

        if state in q_table:
            action = max(q_table[state], key=q_table[state].get)
        else:
            action = random.choice(ACTIONS)

        if action == "move":
            if dist[chosen_lot] == float('inf'):
                return -env.PENALTY - 1

            next_node = env.get_next_node(current, chosen_lot, prev)

            if next_node is None:
                return -env.PENALTY - 2

            edge_weight = None
            for v, w in env.graph[current]:
                if v == next_node:
                    edge_weight = w
                    break

            if edge_weight is None:
                return -env.PENALTY - 3

            level = env.traffic.get((current, next_node), "low")
            factor = env.traffic_multiplier(level)
            step_time = edge_weight * factor
            travel_time += step_time
            current = next_node

        elif action == "switch":
            chosen_lot = random.choice(list(env.LOTS.values()))

        else:
            return env.reward(state, travel_time)

        env.change_environment()

    if not done:
        return -env.PENALTY - 4

def simulate(q_table, num_trials, max_steps, verbose):
    total_score = 0
    success_count = 0

    for i in range(1, num_trials + 1):
        env.create_environment()
        score = run_trial_with_q(q_table, max_steps)
        if score > 0:
            success_count += 1
            total_score += score

        if verbose:
            print(f"Trial {i}: score={score}")

    print(f"Simulated {num_trials} trials")
    print(f"Average score: {total_score / num_trials:.4f}")
    print(f"Success count: {success_count} / {num_trials}")


def main():
    parser = argparse.ArgumentParser(description="Simulate Q-learning using a saved Q-table.")
    parser.add_argument("--q-file", default="q_table.npy", help="Path to the saved Q-table")
    parser.add_argument("--trials", type=int, default=100, help="Number of simulation trials")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Max steps per trial")
    parser.add_argument("--verbose", default=True, help="Print each trial score")
    args = parser.parse_args()

    q_table = load_q_table(args.q_file)
    simulate(q_table, args.trials, args.max_steps, args.verbose)


if __name__ == "__main__":
    main()
