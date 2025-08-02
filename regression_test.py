#!/usr/bin/env python3
import argparse
import fcntl
import math
import os
import pty
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import termios
import time
from pathlib import Path

# Configuration
RENDERER = "./target/release/raytracer"
SCENES_DIR = "scenes"
GOLDEN_DIR = "tests_golden_renders"
SSIM_THRESHOLD = 80.0  # Adjust threshold as needed (higher = stricter)
BUILD_FAILED_CODE = 2
TEST_FAILED_CODE = 1
SCRIPT_ERROR_CODE = 3
USER_INTERRUPTED_CODE = 130

# Command flags
COMMON_FLAGS = ["--depth", "5", "--supersampling-level", "1"]
IMAGE_SPECIFIC_FLAGS = ["--format", "png"]
IMAGE_COMMON_FLAGS = []
ANIMATION_SPECIFIC_FLAGS = ["--fps", "30", "--format", "mp4"]
ANIMATION_COMMON_FLAGS = ["--width", "400", "--height", "400"]


# Colors for output
class Color:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def format_color(text, color_code):
    return f"{color_code}{text}{Color.RESET}"


def main():
    parser = argparse.ArgumentParser(
        description="Regression tests for raytracer renders"
    )
    parser.add_argument("command", choices=["test", "render"], help="Subcommand to run")
    parser.add_argument(
        "--output-dir",
        default="renders",
        help="Output directory for renders (default: renders)",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--show-output",
        action="store_true",
        help="Show renderer output (default)",
    )
    output_group.add_argument(
        "--capture-output",
        action="store_true",
        help="Capture renderer output (do not show)",
    )
    args = parser.parse_args()

    # Default to showing output
    show_output = not args.capture_output
    if args.show_output and args.capture_output:
        print_error("Cannot use --show-output and --capture-output together")
        sys.exit(SCRIPT_ERROR_CODE)

    # Register signal handler for Ctrl-C
    signal.signal(signal.SIGINT, handle_sigint)

    # Initialize summary
    summary = {
        "total": 0,
        "passes": [],
        "render_failures": [],
        "regressions": [],
        "unexpected_failures": [],
        "missing_references": [],
    }

    # Set up output directory
    if args.command == "render":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        skip_comparison = True
        print(f"Rendering all scenes to {output_dir}...")
    else:  # test command
        output_dir = Path(tempfile.mkdtemp())
        skip_comparison = False
        print(f"Temporary output dir: {output_dir}")

    # Build renderer
    if not build_renderer():
        return BUILD_FAILED_CODE

    # Process scenes
    try:
        # Split processing for images and animations
        for scene_type, config in [
            (
                "images",
                {
                    "global_flags": IMAGE_COMMON_FLAGS,
                    "specific_flags": IMAGE_SPECIFIC_FLAGS,
                    "commands": ["image"],
                    "ext": ".png",
                },
            ),
            (
                "animations",
                {
                    "global_flags": ANIMATION_COMMON_FLAGS,
                    "specific_flags": ANIMATION_SPECIFIC_FLAGS,
                    "commands": ["animate"],
                    "ext": ".mp4",
                },
            ),
        ]:
            scene_dir = os.path.join(SCENES_DIR, scene_type)
            scenes = find_scenes(scene_dir)

            for scene_path in scenes:
                summary["total"] += 1
                try:
                    result = run_test(
                        scene_path,
                        scene_type,
                        config,
                        output_dir,
                        skip_comparison,
                        show_output,
                    )
                    full_message = f"{result['scene']}: {result['message']}"
                    if result["type"] == "passed":
                        summary["passes"].append(full_message)
                    elif result["type"] == "render_failure":
                        summary["render_failures"].append(full_message)
                    elif result["type"] == "regression":
                        summary["regressions"].append(full_message)
                    elif result["type"] == "unexpected_failure":
                        summary["unexpected_failures"].append(full_message)
                    elif result["type"] == "missing_reference":
                        summary["missing_references"].append(full_message)
                except Exception as e:
                    # Catch any unexpected errors and log them
                    summary["unexpected_failures"].append(
                        f"Internal script error - {str(e)}"
                    )
                    print_exception(e)

        print_summary(summary)

        # Clean up temporary files
        if not skip_comparison:
            shutil.rmtree(output_dir)

        if summary["render_failures"] or summary["regressions"]:
            return TEST_FAILED_CODE
        return 0

    except Exception as e:
        print_error(f"Script error: {str(e)}")
        return SCRIPT_ERROR_CODE


def handle_sigint(signum, frame):
    """Handle Ctrl-C by exiting gracefully with specific exit code"""
    print("\n" + format_color("Interrupted by user. Exiting gracefully.", Color.YELLOW))
    sys.exit(USER_INTERRUPTED_CODE)


def print_error(message):
    print(format_color(f"Error: {message}", Color.RED))


def print_exception(exception):
    print(
        format_color(
            f"Exception: {type(exception).__name__} - {str(exception)}", Color.RED
        )
    )


def print_warning(message):
    print(format_color(f"Warning: {message}", Color.YELLOW))


def build_renderer(show_output=True):
    """Build the release binary and return success status"""
    print(format_color("Building renderer...", Color.CYAN))
    try:
        cmd = ["cargo", "build", "--release"]
        _, interrupted, returncode = run_command_with_pty(cmd, show_output)
        if interrupted:
            print_warning("Build interrupted by user")
            return False
        elif returncode != 0:
            print_error("Build failed")
            return False
        print(format_color("Build succeeded.", Color.GREEN))
        return True
    except Exception as e:
        print_error(f"Build failed: {str(e)}")
        return False


def run_command_with_pty(cmd, show_output=True):
    # Save cursor position
    sys.stdout.write("\x1b7")
    sys.stdout.flush()

    master_fd, slave_fd = pty.openpty()

    # Set PTY to raw mode
    old_settings = termios.tcgetattr(master_fd)
    new_settings = termios.tcgetattr(master_fd)
    new_settings[3] &= ~termios.ECHO
    termios.tcsetattr(master_fd, termios.TCSANOW, new_settings)

    process = subprocess.Popen(
        cmd,
        stdin=sys.stdin,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    # Make master FD non-blocking
    fl = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    output_bytes = bytearray()
    interrupted = False

    def sigint_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        if process.poll() is None:
            process.send_signal(sig)

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        while True:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if r:
                try:
                    data = os.read(master_fd, 1024)
                    if not data:
                        break
                    output_bytes.extend(data)

                    if show_output:
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                except (BlockingIOError, OSError) as e:
                    if isinstance(e, OSError) and e.errno == 5:  # EIO
                        break
                    continue
            if process.poll() is not None:
                break
    except KeyboardInterrupt:
        interrupted = True
        process.terminate()
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        termios.tcsetattr(master_fd, termios.TCSANOW, old_settings)
        os.close(master_fd)
        process.wait()

        # Clear output only if we *didn’t* show it
        if not show_output:
            sys.stdout.write("\x1b8\x1b[J")
            sys.stdout.flush()

    output_text = output_bytes.decode(errors="replace")
    return output_text, interrupted, process.returncode


def find_scenes(scene_dir):
    """Find YAML scene files in a directory"""
    if not os.path.exists(scene_dir):
        print_warning(f"Scene directory not found: {scene_dir}")
        return []

    scenes = []
    for root, _, files in os.walk(scene_dir):
        for file in files:
            if file.endswith(".yml"):
                scenes.append(os.path.join(root, file))
    return scenes


def render_test(
    cmd,
    scene_name,
    scene_path,
    scene_type,
    config,
    skip_comparison,
    output_rel_path,
    show_output,
):
    print(format_color("\n" + "=" * 50, Color.MAGENTA))
    print(format_color(f"Testing: {scene_path}", Color.BOLD))
    print(format_color(f"Command: {' '.join(cmd)}", Color.CYAN))
    print(format_color("=" * 50, Color.MAGENTA))

    # Compute golden_path and check existence
    golden_path = os.path.join(GOLDEN_DIR, f"{scene_name}{config['ext']}")
    if not os.path.exists(golden_path):
        return {
            "type": "missing_reference",
            "message": "Missing golden reference",
        }

    start_time = time.time()

    output, interrupted, returncode = run_command_with_pty(cmd, show_output)

    elapsed = time.time() - start_time

    # Extract last 10 lines from bytes buffer
    lines = output.splitlines()
    last_10 = lines[-10:] if len(lines) > 10 else lines
    last_10_str = "\n".join(last_10)

    if interrupted:
        return {
            "type": "render_failure",
            "message": "Renderer was interrupted by user",
        }
    if returncode != 0:
        failure_msg = f"Renderer failed. Last 10 lines of output:\n{last_10_str}"
        return {
            "type": "render_failure",
            "message": failure_msg,
        }

    # If rendering only, we're done
    if skip_comparison:
        return {
            "type": "passed",
            "message": f"Rendered in {elapsed:.2f}s",
        }
    # Compare based on scene type
    if scene_type == "images":
        return compare_images(golden_path, output_rel_path, scene_name)

    return compare_animations(golden_path, output_rel_path, scene_name, elapsed)


def run_test(scene_path, scene_type, config, output_dir, skip_comparison, show_output):
    """Run a single test and return result dictionary"""
    # Compute output file path
    scene_name = os.path.splitext(os.path.relpath(scene_path, SCENES_DIR))[0]
    output_rel_path = Path(output_dir) / f"{scene_name}{config['ext']}"
    output_dir_path = output_rel_path.parent
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Build render command with correct argument order:
    #   raytracer [global_flags] [output_dir] <scene_path> [subcommand_global_flags] <subcommand> [subcommand_flags]
    cmd = [
        RENDERER,
        *COMMON_FLAGS,
        "--output-dir",
        str(output_dir_path),
        scene_path,
        *config["global_flags"],
        *config["commands"],
        *config["specific_flags"],
    ]

    try:
        result = render_test(
            cmd,
            scene_name,
            scene_path,
            scene_type,
            config,
            skip_comparison,
            output_rel_path,
            show_output,
        )
        result["scene"] = scene_path

    except Exception as e:
        # Catch errors during rendering or comparison
        failure_msg = f"Unhandled exception for {str(e)}"
        print_exception(e)
        result = {"type": "render_failure", "message": failure_msg, "scene": scene_path}

    # Print a one-liner for the test result
    status_to_color = {
        "passed": Color.GREEN,
        "render_failure": Color.RED,
        "regression": Color.RED,
        "unexpected_failure": Color.RED,
        "missing_reference": Color.YELLOW,
    }
    status_map = {
        "passed": "PASS",
        "render_failure": "FAIL",
        "regression": "REGRESSION",
        "unexpected_failure": "UNEXPECTED FAIL",
        "missing_reference": "MISSING REF",
    }
    if result is not None:
        status_str = status_map[result["type"]]
        color = status_to_color[result["type"]]
        message = result["message"]
        if show_output and result["type"] == "render_failure":
            # if the renderer failed, and the output was shown, don't reprint the errors now
            message = "Renderer failed"
        print(format_color(f"[{status_str}] {scene_name}: {message}", color))

    print(format_color("-" * 50, Color.MAGENTA))
    # Reset interrupted flag after handling
    return result


def compare_images(golden_path, test_path, scene_name):
    """Compare two images using ssimulacra2_rs"""
    try:
        cmd = ["ssimulacra2_rs", "image", golden_path, str(test_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return {
                "type": "regression",
                "message": f"Comparison failed (exit code {result.returncode})",
                "scene": scene_name,
            }

        # Parse SSIM score from output
        output = result.stdout.strip()
        try:
            score = float(output.split()[-1])
        except (IndexError, ValueError):
            return {
                "type": "regression",
                "message": "Failed to parse SSIM score",
                "scene": scene_name,
            }

        # Evaluate against threshold
        if score >= SSIM_THRESHOLD:
            return {
                "type": "passed",
                "message": f"Image pass with SSIM: {score}",
                "scene": scene_name,
            }
        else:
            return {
                "type": "regression",
                "message": f"Difference exceeds threshold (SSIM={score:.1f} < {SSIM_THRESHOLD})",
                "scene": scene_name,
            }

    except FileNotFoundError:
        return {
            "type": "regression",
            "message": "Comparison tool missing",
            "scene": scene_name,
        }


def compare_animations(golden_path, test_path, scene_name, elapsed_time):
    """Compare two animations by extracting frames"""
    try:
        # Create temporary directories
        with (
            tempfile.TemporaryDirectory() as ref_dir,
            tempfile.TemporaryDirectory() as test_dir,
        ):
            # Extract frames
            if not extract_video_frames(golden_path, ref_dir):
                return {
                    "type": "unexpected_failure",
                    "message": "Failed to extract reference frames",
                    "scene": scene_name,
                }

            if not extract_video_frames(test_path, test_dir):
                return {
                    "type": "unexpected_failure",
                    "message": "Failed to extract test frames",
                    "scene": scene_name,
                }

            # Get frame lists
            ref_frames = sorted(os.listdir(ref_dir))
            test_frames = sorted(os.listdir(test_dir))

            # Verify frame counts
            if len(ref_frames) != len(test_frames):
                return {
                    "type": "regression",
                    "message": (
                        f" Frame count mismatch: "
                        f"ref={len(ref_frames)} vs test={len(test_frames)}"
                    ),
                    "scene": scene_name,
                }

            # Find frame with highest difference
            worst_frame = None
            worst_score = float("inf")

            for frame in ref_frames:
                ref_path = os.path.join(ref_dir, frame)
                test_path = os.path.join(test_dir, frame)

                # For each frame, compute SSIM
                score = get_frame_score(ref_path, test_path)
                if math.isinf(score):
                    return {
                        "type": "unexpected_failure",
                        "message": "Failed to compare frame {frame}",
                        "scene": scene_name,
                    }

                if score < worst_score:
                    worst_score = score
                    worst_frame = frame

            # Evaluate worst frame against threshold
            if worst_score >= SSIM_THRESHOLD:
                return {
                    "type": "passed",
                    "message": (
                        f"Animation passed in {elapsed_time:.2f}s. "
                        f"Worst frame: {worst_frame} ({worst_score})"
                    ),
                    "scene": scene_name,
                }
            else:
                return {
                    "type": "regression",
                    "message": (
                        f"Difference in frame {worst_frame} exceeds threshold "
                        f"(SSIM={worst_score:.1f} < {SSIM_THRESHOLD})"
                    ),
                    "scene": scene_name,
                }

    except Exception as e:
        return {
            "type": "regression",
            "message": f"Animation comparison error: {str(e)}",
            "scene": scene_name,
        }


def extract_video_frames(video_path, output_dir):
    """Extract video frames using ffmpeg"""
    try:
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-loglevel",
            "error",
            os.path.join(output_dir, "frame_%04d.png"),
        ]
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_frame_score(ref_path, test_path):
    """Get SSIM score for a single frame"""
    try:
        cmd = ["ssimulacra2_rs", "image", ref_path, test_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return float("inf")

        output = result.stdout.strip()
        try:
            return float(output.split()[-1])
        except (IndexError, ValueError):
            return float("inf")

    except FileNotFoundError:
        return float("inf")


def print_summary(summary):
    """Print test summary with details"""
    print("\n" + format_color("=" * 50, Color.MAGENTA))
    print(format_color(f"{'Test Summary':^50}", Color.MAGENTA + Color.BOLD))
    print(format_color("=" * 50, Color.MAGENTA))

    total = summary["total"]
    passed = len(summary["passes"])
    render_failures = len(summary["render_failures"])
    unexpected_failures = len(summary["unexpected_failures"])
    regressions = len(summary["regressions"])
    missing_references = len(summary["missing_references"])

    print(f"Total tests:      {format_color(str(total), Color.CYAN)}")
    print(f"Passed tests:     {format_color(str(passed), Color.GREEN)}")
    print(f"Render fails:     {format_color(str(render_failures), Color.RED)}")
    print(f"Regressions:      {format_color(str(regressions), Color.RED)}")
    print(f"Unexpected fails: {format_color(str(unexpected_failures), Color.RED)}")
    print(f"Missing refs:     {format_color(str(missing_references), Color.YELLOW)}")

    # Print passes if any
    if passed > 0:
        print("\n" + format_color("Passes:", Color.GREEN + Color.BOLD))
        for _pass in summary["passes"]:
            print(f"  • {_pass}")
    else:
        print(format_color("No tests passed.", Color.YELLOW + Color.BOLD))

    # Print render failures if any
    if render_failures > 0:
        print("\n" + format_color("Render Failures:", Color.RED + Color.BOLD))
        for failure in summary["render_failures"]:
            print(f"  • {failure}")

    # Print regressions if any
    if regressions > 0:
        print("\n" + format_color("Regressions:", Color.RED + Color.BOLD))
        for regression in summary["regressions"]:
            print(f"  • {regression}")

    # Print missing references if any
    if unexpected_failures > 0:
        print("\n" + format_color("Unexpected failures:", Color.RED + Color.BOLD))
        for fail in summary["unexpected_failures"]:
            print(f"  • {fail}")

    # Print missing references if any
    if missing_references > 0:
        print("\n" + format_color("Missing References:", Color.YELLOW + Color.BOLD))
        for ref in summary["missing_references"]:
            print(f"  • {ref}")

    print(format_color("=" * 50, Color.MAGENTA))

    # Final status
    if passed == total:
        final_msg = "ALL TESTS PASSED"
        color = Color.GREEN
    else:
        final_msg = "TESTS FAILED"
        color = Color.RED

    print(format_color(f"\n{final_msg}", color + Color.BOLD))
    print(format_color("=" * 50, Color.MAGENTA))


if __name__ == "__main__":
    sys.exit(main())
