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

The output file for each graph uses CSV formatting so it can be imported into Excel.

- First line contains the headers
- Remaining lines contain the data rows

## Run `q_l_scale_compare.py`

Use the following command:

```bash
python q_l_scale_compare.py <graph_file> <output_file>
```

## Parallel Training

For different graph files, fill in `parallel_input.txt` with this format:

```text
<number of test sets>

<graph_file_1> <output_file_1>
...
<graph_file_n> <output_file_n>
```

Then compile and run `parallel_training.py` to generate the corresponding output files.

## Important

- Parallel training uses multiprocessing.
- Be careful with the number of test sets.
