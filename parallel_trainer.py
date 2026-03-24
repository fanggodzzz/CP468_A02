import os
import subprocess
from multiprocessing import Pool, cpu_count

INPUT_FILE = "input.txt"


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
        raise ValueError("First line of input file must be an integer (number of trainers)") from ex

    if n < 1:
        raise ValueError("Number of training jobs must be at least 1")

    if len(lines) < n + 1:
        raise ValueError(f"Input file must contain {n} parameter lines after the first line")

    configs = []
    for i in range(1, n + 1):
        parts = lines[i].split()
        if len(parts) != 5:
            raise ValueError(f"Line {i+1} must contain 4 values: ALPHA ALPHA_DECAY EPSILON EPSILON_DECAY")

        alpha, alpha_decay, epsilon, epsilon_decay, decay_period = parts
        configs.append((i, float(alpha), float(alpha_decay), float(epsilon), float(epsilon_decay), int(decay_period)))

    return configs


def run_training_job(args):
    # args = (idx, alpha, alpha_decay, epsilon, epsilon_decay, decay_period)
    idx, alpha, alpha_decay, epsilon, epsilon_decay, decay_period = args
    output_file = f"OUTPUT_{idx}.txt"
    cmd = [
        "python",
        "q_learning_parking1.py",
        str(alpha),
        str(alpha_decay),
        str(epsilon),
        str(epsilon_decay),
        str(decay_period),
        output_file
    ]

    print(f"[Job {idx}] Starting: {cmd}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        with open(output_file, "a", encoding="utf-8") as f:
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
            print(f"[Job {idx}] Completed successfully and output saved to {output_file}")
        else:
            print(f"[Job {idx}] Failed with return code {result.returncode}; see {output_file}")

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