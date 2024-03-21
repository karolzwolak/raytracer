from datetime import datetime
import sys
import subprocess

BUILD_COMMAND = "cargo build -r"
TARGET_DIR = "target/release/raytracer"
RUN_COMMAND = f"{TARGET_DIR} "
DEFAULT_TIMES = 2


def build():
    return subprocess.run(BUILD_COMMAND, shell=True)


def run(args=""):
    subprocess.run(RUN_COMMAND + args, shell=True)


def time_run(args=""):
    now = datetime.now()
    run(args)
    print()
    elapsed = datetime.now() - now
    return elapsed.total_seconds()


def run_n_times(n, args=""):
    if n < 1:
        return 0, 0
    time = 0
    for _ in range(n):
        time += time_run(args)
    return round(time, 3), round(time / n, 3)


def main():
    build()

    args = sys.argv[1:]

    times = int(args[0]) if args else DEFAULT_TIMES
    run_args = " ".join(args[1:]) if args else ""
    time, avg = run_n_times(times, run_args)

    print(f"ran {times} times")
    print(f"total time: {time} seconds")
    print(f"average time: {avg} seconds")


if __name__ == "__main__":
    main()
