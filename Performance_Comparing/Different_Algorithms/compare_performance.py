import argparse
import numpy as np
import random
import os

import parking_env as env

MAX_STEPS = 100
NUM_SAMPLES = 10000
Q = {}
OUTPUT_FILE = os.path.join(env.script_dir, "output.txt")


def write_result(msg):
    with open(OUTPUT_FILE, "w") as f:
        f.write(msg + "\n")

def load_q_table(filename):
    data = np.load(filename, allow_pickle=True)
    if isinstance(data, np.ndarray):
        return data.item()
    return data

def make_agent_invalid(agent):
    agent["done"] = True
    agent["steps"] = MAX_STEPS + 10


def parse_bool(value):
    if isinstance(value, bool):
        return value

    lowered = value.strip().lower()
    if lowered in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "f", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError("verbose must be a boolean value such as True or False")

def run_q_learning_step(agent, q_table):
    current = agent["current"]
    chosen_lot = agent["chosen_lot"]

    dist, prev = env.Dijkstra(current)
    state = env.create_state(current, chosen_lot, dist, prev)

    if state not in q_table:
        msg = "ERROR: Not trained situation"
        agent["actions"].append(msg)
        make_agent_invalid(agent)
        return
    
    action = max(q_table[state], key=q_table[state].get)
    if action == "move":
        if current == chosen_lot or dist[chosen_lot] == float('inf'):
            msg = "ERROR: Move at parking lot"
            agent["actions"].append(msg)
            make_agent_invalid(agent)
            return
        
        next_node = env.get_next_node(current, chosen_lot, prev)
        if next_node is None:
            msg = "ERROR: No path to chosen lot"
            agent["actions"].append(msg)
            make_agent_invalid(agent)
            return
        
        edge_weight = next((w for v, w in env.graph[current] if v == next_node), None)
        level = env.traffic.get((current, next_node), "low")
        factor = env.traffic_multiplier(level)
        step_time = edge_weight * factor
        agent["travel_time"] += step_time
        agent["current"] = next_node
        
        msg = f"MOVE: {current} --> {next_node}, Time travel: {step_time}"
        agent["actions"].append(msg)

    elif action == "switch":
        temp = agent["chosen_lot"]
        best_lot = env.find_the_best_lot_by_distance(current, dist)
        if best_lot is None:
            best_lot = random.choice(env.PARKING_LOTS)
        agent["chosen_lot"] = best_lot
        msg = f"SWITCH LOT: {temp} --> {agent['chosen_lot']}"
        agent["actions"].append(msg)

    else:
        ans = env.reward(state, agent["travel_time"])
        if (ans < 0):
            msg = f"PARKING UNSUCCESSFUL: {current}, Time travel: {agent['travel_time']}, Score: {ans}"
            agent["actions"].append(msg)
            make_agent_invalid(agent)
            return
        else :
            msg = f"PARKING SUCCESSFUL: {current}, Time travel: {agent['travel_time']}, Score: {ans}"
            agent["actions"].append(msg)
            agent["score"] = ans
            agent["done"] = True

    agent["steps"] += 1

def run_greedy_step(agent, dist, prev):
    current = agent["current"]
    chosen_lot = agent["chosen_lot"]

    if chosen_lot is None:
        msg = "ERROR: No chosen lot for Dijkstra agent, stay still at node 0"
        agent["actions"].append(msg)
        chosen_lot = env.find_the_best_lot_by_distance(0, dist)
        if chosen_lot is not None:
            msg = f"START: Chosen lot {chosen_lot}"
            agent["actions"].append(msg)
            agent["chosen_lot"] = chosen_lot
        return

    if current == chosen_lot:
        ans = env.reward1(current, chosen_lot, agent["travel_time"])
        if (ans < 0):
            msg = f"PARKING UNSUCCESSFUL: {current}, Time travel: {agent['travel_time']}, Score: {ans}"
            agent["actions"].append(msg)
            make_agent_invalid(agent)
        else :
            msg = f"PARKING SUCCESSFUL: {current}, Time travel: {agent['travel_time']}, Score: {ans}"
            agent["actions"].append(msg)
            agent["score"] = ans
            agent["done"] = True
        agent["steps"] += 1
        return

    next_node = env.get_next_node(current, chosen_lot, prev)
    if next_node is None:
        msg = f"ERROR: No path to best lot {chosen_lot}"
        agent["actions"].append(msg)
        make_agent_invalid(agent)
        return

    edge_weight = next((w for v, w in env.graph[current] if v == next_node), None)

    level = env.traffic.get((current, next_node), "low")
    factor = env.traffic_multiplier(level)
    step_time = edge_weight * factor
    agent["travel_time"] += step_time
    agent["current"] = next_node
    
    msg = f"MOVE: {current} --> {next_node}, Time travel: {step_time}"
    agent["actions"].append(msg)
    agent["steps"] += 1


def run_dijkstra_step(agent, dist, prev):
    current = agent["current"]
    chosen_lot = agent["chosen_lot"]

    if chosen_lot is None:
        msg = "ERROR: No chosen lot for Dijkstra agent, stay still at node 0"
        agent["actions"].append(msg)
        chosen_lot = env.find_the_best_lot_by_distance(0, dist)
        if chosen_lot is not None:
            msg = f"START: Chosen lot {chosen_lot}"
            agent["actions"].append(msg)
            agent["chosen_lot"] = chosen_lot
        return

    if current == chosen_lot:
        ans = env.reward1(current, chosen_lot, agent["travel_time"])
        if (ans < 0):
            msg = f"PARKING UNSUCCESSFUL: {current}, Time travel: {agent['travel_time']}, Score: {ans}"
            agent["actions"].append(msg)
            make_agent_invalid(agent)
        else :
            msg = f"PARKING SUCCESSFUL: {current}, Time travel: {agent['travel_time']}, Score: {ans}"
            agent["actions"].append(msg)
            agent["score"] = ans
            agent["done"] = True
        agent["steps"] += 1
        return

    next_node = env.get_next_node(current, chosen_lot, prev)
    if next_node is None:
        msg = f"ERROR: No path to best lot {chosen_lot}"
        agent["actions"].append(msg)
        make_agent_invalid(agent)
        return

    edge_weight = next((w for v, w in env.graph[current] if v == next_node), None)

    level = env.traffic.get((current, next_node), "low")
    factor = env.traffic_multiplier(level)
    step_time = edge_weight * factor
    agent["travel_time"] += step_time
    agent["current"] = next_node
    
    msg = f"MOVE: {current} --> {next_node}, Time travel: {step_time}"
    agent["actions"].append(msg)
    agent["steps"] += 1


def run_three_methods_simultaneously(q_table):

    agents = [
        {
            "name": "q",
            "current": 0,
            "chosen_lot": random.choice(env.PARKING_LOTS),
            "travel_time": 0,
            "score": 0,
            "steps": 0,
            "done": False,
            "actions": []
        },
        {
            "name": "greedy",
            "current": 0,
            "chosen_lot": None,
            "travel_time": 0,
            "score": 0,
            "steps": 0,
            "done": False,
            "actions": []
        },
        {
            "name": "dijkstra",
            "current": 0,
            "chosen_lot": None,
            "travel_time": 0,
            "score": 0,
            "steps": 0,
            "done": False,
            "actions": []
        },
    ]

    env.change_environment()
    dist, prev = env.Dijkstra(0)
    agents[1]["chosen_lot"] = env.find_the_best_lot_by_cost(0, dist)
    agents[2]["chosen_lot"] = env.find_the_best_lot_by_distance(0, dist)
    while any(not a["done"] and a["steps"] < MAX_STEPS for a in agents):
        for agent in agents:
            if agent["done"] or agent["steps"] >= MAX_STEPS:
                continue
            if agent["name"] == "q":
                run_q_learning_step(agent, q_table)
            elif agent["name"] == "greedy":
                run_greedy_step(agent, dist, prev)
            else:
                run_dijkstra_step(agent, dist, prev)

        env.change_environment()
    
    for agent in agents:
        if agent["steps"] > MAX_STEPS:
            agent["score"] = -1
    return {agent["name"]: agent["score"] for agent in agents}, {agent["name"]: agent["actions"] for agent in agents}


def compare(q_file, samples, verbose):
    q_table = load_q_table(q_file)
    total = {"q": 0, "greedy": 0, "dijkstra": 0}
    success = {"q": 0, "greedy": 0, "dijkstra": 0}
    action_sequence = {"q":[], "greedy":[], "dijkstra":[]}

    for i in range(1, samples + 1):
        msg = f"Try {i} / {samples}"
        print(msg, end='\r')
        env.create_environment()
        scores, action_sequence = run_three_methods_simultaneously(q_table)

        q_score = scores["q"]
        greedy_score = scores["greedy"]
        dijkstra_score = scores["dijkstra"]

        total["q"] += q_score if q_score >= 0 else 0
        total["greedy"] += greedy_score if greedy_score >= 0 else 0
        total["dijkstra"] += dijkstra_score if dijkstra_score >= 0 else 0

        success["q"] += 1 if q_score >= 0 else 0
        success["greedy"] += 1 if greedy_score >= 0 else 0
        success["dijkstra"] += 1 if dijkstra_score >= 0 else 0

        if verbose:
            for algo, actions in action_sequence.items():
                print(f"---Actions sequence for {algo}:")
                for action in actions:
                    print(action)
                print()
            print(
                f"Sample {i}: q={q_score}, greedy={greedy_score}, dijkstra={dijkstra_score}"
            )
    
    msg = f"q-learning avg score, q-learning success, "
    msg += f"greedy avg score, greedy success, "
    msg += f"dijkstra avg score, dijkstra success, "
    msg += "samples\n"

    msg += f"{total['q'] / samples:.4f}, {success['q']}, "
    msg += f"{total['greedy'] / samples:.4f}, {success['greedy']}, "
    msg += f" {total['dijkstra'] / samples:.4f}, {success['dijkstra']}"
    msg += f", {samples}"
    write_result(msg)

    print(f"Completed {samples} samples")
    print(f"Average Q-learning score: {total['q'] / samples:.4f}, success {success['q']} / {samples}")
    print(f"Average greedy score: {total['greedy'] / samples:.4f}, success {success['greedy']} / {samples}")
    print(f"Average dijkstra score: {total['dijkstra'] / samples:.4f}, success {success['dijkstra']} / {samples}")


def main():
    parser = argparse.ArgumentParser(description="Compare Q-learning, greedy cost, and Dijkstra methods.")
    parser.add_argument("q_file", help="Path to the saved Q-table")
    parser.add_argument("samples", type=int, help="Number of samples to compare")
    parser.add_argument("verbose", type=parse_bool, help="Print every sample result")
    args = parser.parse_args()

    compare(args.q_file, args.samples, args.verbose)


if __name__ == "__main__":
    main()
