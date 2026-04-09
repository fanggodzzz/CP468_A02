import argparse
import numpy as np
import random

import parking_env as env

MAX_STEPS = 100
NUM_SAMPLES = 10000


def load_q_table(filename):
    data = np.load(filename, allow_pickle=True)
    if isinstance(data, np.ndarray):
        return data.item()
    return data


def run_q_learning_method(q_table, max_steps):
    current = 0
    chosen_lot = random.choice(list(env.LOTS.values()))
    travel_time = 0
    steps = 0
    done = False

    env.change_environment()

    while not done and steps < max_steps:
        steps += 1
        dist, prev = env.Dijkstra(current)
        state = (current, chosen_lot, tuple(env.p_avail))

        if state in q_table:
            action = max(q_table[state], key=q_table[state].get)
        else:
            action = random.choice(["move", "switch", "park"])

        if action == "move":
            if current == chosen_lot:
                r = -5
                next_state = (current, chosen_lot, tuple(env.p_avail))
                done = False
            elif dist[chosen_lot] == float('inf'):
                r = -100
                next_state = (current, chosen_lot, tuple(env.p_avail))
                done = False
            else:
                next_node = env.get_next_node(current, chosen_lot, prev)
                edge_weight = None
                for v, w in env.graph[current]:
                    if v == next_node:
                        edge_weight = w
                        break

                if edge_weight is None:
                    r = -100
                    next_state = (current, chosen_lot, tuple(env.p_avail))
                    done = False
                else:
                    level = env.traffic.get((current, next_node), "low")
                    factor = env.traffic_multiplier(level)
                    step_time = edge_weight * factor
                    travel_time += step_time
                    next_state = (next_node, chosen_lot, tuple(env.p_avail))
                    r = -step_time
                    done = False

        elif action == "switch":
            chosen_lot = random.choice(list(env.LOTS.values()))
            next_state = (current, chosen_lot, tuple(env.p_avail))
            r = -5
            done = False

        else:
            next_state = (current, chosen_lot, tuple(env.p_avail))
            if current != chosen_lot:
                r = -100
                done = False
            else:
                r = env.reward(next_state, travel_time)
                done = True

        current = next_state[0]
        env.change_environment()

    return r if done else 0


def run_greedy_cost_method(max_steps):
    current = 0
    travel_time = 0
    steps = 0
    done = False

    env.change_environment()

    while not done and steps < max_steps:
        steps += 1
        dist, prev = env.Dijkstra(current)

        best_lot = None
        best_cost = float('inf')

        for lot in env.LOTS.values():
            idx = env.LOT_INDEX[lot]
            if env.p_avail[idx] == 0 or env.walk_km[idx] > env.W_MAX:
                continue
            if dist[lot] == float('inf'):
                continue

            cost = (
                dist[lot]
                + env.LAMBDA_WALK * env.walk_km[idx]
                + env.LAMBDA_PRICE * env.price[idx]
            )
            if cost < best_cost:
                best_cost = cost
                best_lot = lot

        if best_lot is None:
            return 0

        if current == best_lot:
            r = env.reward((current, best_lot, tuple(env.p_avail)), travel_time)
            done = True
            break

        next_node = env.get_next_node(current, best_lot, prev)
        if next_node is None:
            return 0

        edge_weight = None
        for v, w in env.graph[current]:
            if v == next_node:
                edge_weight = w
                break

        if edge_weight is None:
            return 0

        level = env.traffic.get((current, next_node), "low")
        factor = env.traffic_multiplier(level)
        step_time = edge_weight * factor
        travel_time += step_time
        current = next_node
        env.change_environment()

    return r if done else 0


def run_dijkstra_method(max_steps):
    current = 0
    travel_time = 0
    steps = 0
    done = False

    env.change_environment()

    while not done and steps < max_steps:
        steps += 1
        dist, prev = env.Dijkstra(current)

        best_lot = None
        best_dist = float('inf')

        for lot in env.LOTS.values():
            idx = env.LOT_INDEX[lot]
            if env.p_avail[idx] == 0 or env.walk_km[idx] > env.W_MAX:
                continue
            if dist[lot] < best_dist:
                best_dist = dist[lot]
                best_lot = lot

        if best_lot is None:
            return 0

        if current == best_lot:
            r = env.reward((current, best_lot, tuple(env.p_avail)), travel_time)
            done = True
            break

        next_node = env.get_next_node(current, best_lot, prev)
        if next_node is None:
            return 0

        edge_weight = None
        for v, w in env.graph[current]:
            if v == next_node:
                edge_weight = w
                break

        if edge_weight is None:
            return 0

        level = env.traffic.get((current, next_node), "low")
        factor = env.traffic_multiplier(level)
        step_time = edge_weight * factor
        travel_time += step_time
        current = next_node
        env.change_environment()

    return r if done else 0


def compare(q_file, samples, max_steps, verbose):
    q_table = load_q_table(q_file)
    total = {"q": 0, "greedy": 0, "dijkstra": 0}
    success = {"q": 0, "greedy": 0, "dijkstra": 0}

    for i in range(1, samples + 1):
        env.create_environment()
        snapshot = env.snapshot_environment()

        q_score = run_q_learning_method(q_table, max_steps)
        env.restore_environment(snapshot)
        greedy_score = run_greedy_cost_method(max_steps)
        env.restore_environment(snapshot)
        dijkstra_score = run_dijkstra_method(max_steps)

        total["q"] += q_score
        total["greedy"] += greedy_score
        total["dijkstra"] += dijkstra_score

        success["q"] += 1 if q_score > 0 else 0
        success["greedy"] += 1 if greedy_score > 0 else 0
        success["dijkstra"] += 1 if dijkstra_score > 0 else 0

        if verbose:
            print(
                f"Sample {i}: q={q_score}, greedy={greedy_score}, dijkstra={dijkstra_score}"
            )

    print(f"Completed {samples} samples")
    print(f"Average Q-learning score: {total['q'] / samples:.4f}, success {success['q']} / {samples}")
    print(f"Average greedy score: {total['greedy'] / samples:.4f}, success {success['greedy']} / {samples}")
    print(f"Average dijkstra score: {total['dijkstra'] / samples:.4f}, success {success['dijkstra']} / {samples}")


def main():
    parser = argparse.ArgumentParser(description="Compare Q-learning, greedy cost, and Dijkstra methods.")
    parser.add_argument("--q-file", default="q_table.npy", help="Path to the saved Q-table")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES, help="Number of samples to compare")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Max steps per sample")
    parser.add_argument("--verbose", action="store_true", help="Print every sample result")
    args = parser.parse_args()

    compare(args.q_file, args.samples, args.max_steps, args.verbose)


if __name__ == "__main__":
    main()
