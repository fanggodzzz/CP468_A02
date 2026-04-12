# Instructions

## Input File Format

Graph input files use a weighted, undirected graph. The driver starts at node `0`.

### Notes

- `n`: number of vertices
- `m`: number of edges
- `p`: number of parking lots
- `in`: number of invalid parking lots, meaning lots farther than the maximum walking distance
- `a1 a2 ... ap`: list of parking lot nodes
- `b1 b2 ... bp`: list of invalid parking nodes
- `u v w`: edge between `u` and `v` with weight `w`

### Format

```text
n m p in
a1 a2 ... ap
b1 b2 ... bp
u1 v1 w1
...
um vm wm
```

## Output File Format

The output file produced by `compare_performance.py` uses CSV formatting so it can be imported into Excel.

- First line contains the headers
- Remaining lines contain the data rows

## Run Order

Compile in this order to produce the intended output file:

1. `training.py`
2. `compare_performance.py`

## Training

Use the following command:

```bash
python training.py <graph_file> <episodes>
```

The included `q_table.npy` file is the result of training for 200000 episodes.

## Performance Comparison

Use the following command:

```bash
python compare_performance.py <q_file> <number_of_samples> <whether_to_print_trials>
```

The output file is `output.txt` in CSV form.
