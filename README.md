# CP468_A02

Assignment 2 for CP468. This repository contains a Q-learning parking-lot simulation, comparison scripts for multiple decision strategies, and scaling experiments for different graph inputs.

## Project Layout

- `q_learning_parking_original.py`: original single-script version of the parking environment and Q-learning logic.
- `Performance_Comparing/Different_Algorithms/`: compares Q-learning, greedy cost, and Dijkstra-based parking selection.
- `Performance_Comparing/Graph_scale/`: runs scaling experiments across multiple graph files.
- `graph.txt`: sample graph input used by the root-level scripts.

## Requirements

- Python 3
- NumPy

## Graph File Format

Graph inputs use a weighted, undirected graph format. The driver starts at node `0`.

```text
n m p in
a1 a2 ... ap
b1 b2 ... bin
u1 v1 w1
...
um vm wm
```

Where:

- `n` is the number of vertices
- `m` is the number of edges
- `p` is the number of parking lots
- `in` is the number of invalid parking lots
- `a1 ... ap` lists the parking lot nodes
- `b1 ... bin` lists the invalid parking nodes
- `u v w` defines an undirected edge between `u` and `v` with weight `w`

## Different Algorithms Comparison

From `Performance_Comparing/Different_Algorithms/`:

1. Train the Q-table:

   ```bash
   python training.py <graph_file> <episodes>
   ```

2. Compare the methods:

   ```bash
   python compare_performance.py <q_file> <number_of_samples> <whether_to_print_trials>
   ```

The comparison script writes its results to `output.txt` in CSV form.

## Graph Scale Experiments

From `Performance_Comparing/Graph_scale/`:

1. Run a single comparison:

   ```bash
   python q_l_scale_compare.py <graph_file> <output_file>
   ```

2. Run batch experiments with multiprocessing:
   - Fill in `parallel_input.txt`
   - Run `parallel_training.py`

The batch script reads the test set list from `parallel_input.txt` and produces the matching output files.

## Notes

- The repository already includes sample output files under each experiment folder.
- `q_table.npy` is the saved Q-table produced by training.
