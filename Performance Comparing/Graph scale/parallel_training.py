import os
import sys
import subprocess
from multiprocessing import Pool, cpu_count

script_dir = os.path.dirname(__file__)
INPUT_FILE = os.path.join(script_dir, "parallel_input.txt")


def read_training_config(input_file=INPUT_FILE):
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("Input file is empty")

    try:
        n = int(lines[0])
    except ValueError as ex:
        raise ValueError("First line of input file must be an integer (number of test sets)") from ex

    if n < 1:
        raise ValueError("Number of training jobs must be at least 1")

    if len(lines) < n + 1:
        raise ValueError(f"Input file must contain {n} graph/output lines after the first line")

    configs = []
    for i in range(1, n + 1):
        parts = lines[i].split()
        if len(parts) != 2:
            raise ValueError(f"Line {i+1} must contain 2 values: graph_file output_file")

        graph_file, output_file = parts
        configs.append((i, graph_file, output_file))

    return configs


def run_training_job(args):
    idx, graph_file, output_file = args
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        "q_l_scale_compare.py",
        graph_file,
        output_file,
    ]

    print(f"[Job {idx}] Starting: {cmd}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)

        output_path = os.path.join(script_dir, output_file)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write("=" * 70 + "\n")
            f.write(f"Job {idx} completed with return code {result.returncode}\n")
            f.write("STDOUT:\n")
            f.write(result.stdout + "\n")
            if result.stderr:
                f.write("STDERR:\n")
                f.write(result.stderr + "\n")
            f.write("=" * 70 + "\n")

        if result.returncode == 0:
            print(f"[Job {idx}] Completed successfully and output saved to {output_path}")
        else:
            print(f"[Job {idx}] Failed with return code {result.returncode}; see {output_path}")

        return idx, result.returncode == 0

    except Exception as e:
        print(f"[Job {idx}] ERROR: {e}")
        return idx, False


def main():
    configs = read_training_config(INPUT_FILE)

    n_jobs = len(configs)
    pool_size = min(n_jobs, cpu_count())

    print("=" * 70)
    print(f"Starting {n_jobs} training jobs on {pool_size} worker processes")
    print("=" * 70)

    with Pool(processes=pool_size) as pool:
        pool.map(run_training_job, configs)

    print("=" * 70)
    print(f"Finished {n_jobs} jobs")
    print("=" * 70)


if __name__ == "__main__":
    main()
