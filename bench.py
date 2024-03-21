from datetime import datetime
import sys
import subprocess

BUILD_COMMAND = "cargo build -r"
TARGET_DIR = "target/release/raytracer"
RUN_COMMAND = f"{TARGET_DIR} "


def build():
    return subprocess.run(BUILD_COMMAND, shell=True)


def run(args=""):
    subprocess.run(RUN_COMMAND + args, shell=True)


def time_run(args=""):
    now = datetime.now()
    run(args)
    elapsed = datetime.now() - now
    return elapsed.total_seconds()


def run_n_times(n, args=""):
    time = 0
    for _ in range(n):
        time += time_run(args)
    return time, time / n


def main():
    build()

    args = sys.argv[1:]
    times = int(args[0])
    args = " ".join(args[1:]) if args else ""
    time, avg = run_n_times(times, args)

    print()
    print(f"Total time: {time} seconds")
    print(f"Average time: {avg} seconds")


if __name__ == "__main__":
    main()
