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
SHOWCASE_DIR = "showcase_renders"
SSIM_THRESHOLD = 80.0
BUILD_FAILED_CODE = 2
TEST_FAILED_CODE = 1
SCRIPT_ERROR_CODE = 3
USER_INTERRUPTED_CODE = 130

TEST_CONFIGS = {
    "general": {
        "--depth": "5",
        "--supersampling-level": "1",
    },
    "image": {
        "commands": ["image"],
        "ext": ".png",
        "subcommand_flags": {"--format": "png"},
    },
    "animation": {
        "commands": ["animate"],
        "ext": ".mp4",
        "subcommand_flags": {"--fps": "30", "--format": "mp4"},
        "general_flags": {"--width": "400", "--height": "400"},
    },
}

SHOWCASE_CONFIGS = {
    "general": {
        "--depth": "5",
        "--supersampling-level": "2",
        "--width": "1200",
        "--height": "1200",
    },
    "image": {
        "commands": ["image"],
        "ext": ".png",
        "subcommand_flags": {"--format": "png"},
    },
    "animation": {
        "commands": ["animate"],
        "ext": ".webp",
        "subcommand_flags": {"--format": "webp", "--fps": "60"},
    },
}

# Define render presets - combines general with type-specific flags
RENDER_PRESETS = {
    "test": TEST_CONFIGS,
    "showcase": SHOWCASE_CONFIGS,
}

# Define showcase scenes - scenes that receive special treatment
SHOWCASE_SCENES = {
    "images/general/cover.yml": None,
    "images/general/dragons.yml": {
        "general": {"--width": "1200", "--height": "800"},
        "image": {"subcommand_flags": {"--format": "png"}},
    },
    "images/chapters/cubes.yml": None,
    "images/chapters/refractions.yml": None,
    "animations/general/csg.yml": {
        "general": {"--width": "800", "--height": "800"},
    },
    "animations/general/rotating_dragon.yml": {
        "general": {"--width": "1200", "--height": "800"},
        "animation": {"subcommand_flags": {"--format": "webp", "--fps": "60"}},
    },
}


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


def record_result(result, summary):
    """Record a test result in the summary dictionary"""
    if "scene" not in result:
        return

    result_type = result.get("type", "unexpected_failure")

    # Store the result in the summary
    summary["results"].append(
        {"type": result_type, "scene": result["scene"], "message": result["message"]}
    )

    # Print one-liner result
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

    status_str = status_map.get(result_type, "UNKNOWN STATUS")
    color = status_to_color.get(result_type, Color.RED)

    print(format_color(f"[{status_str}] {result['scene']}: {result['message']}", color))


def compare_output(scene_group, golden_path, test_path, elapsed_time=None):
    """Compare rendered output with golden reference based on scene group"""
    if scene_group == "image":
        return compare_images(golden_path, test_path, elapsed_time)
    elif scene_group == "animation":
        return compare_animations(golden_path, test_path, elapsed_time)
    return {
        "type": "unexpected_failure",
        "message": f"Unknown scene group for comparison: {scene_group}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Regression tests for raytracer renders"
    )
    parser.add_argument("command", choices=["test", "render"], help="Subcommand to run")
    parser.add_argument(
        "--output-dir",
        default=GOLDEN_DIR,
        help="Output directory for renders (default: renders)",
    )
    parser.add_argument(
        "scenes",
        metavar="SCENE_FILE",
        nargs="*",
        help="Optional list of scene files to test (must be under scenes/images or scenes/animations)",
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
    parser.add_argument(
        "--hide-progress-bars",
        action="store_true",
        help="Hide progress bars in renderer output",
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
        "results": [],
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
    if not build_renderer(show_output):
        return BUILD_FAILED_CODE

    # Get list of scenes if provided by user
    scene_list = []
    scene_groups = ["images", "animations"]

    if args.scenes:
        for scene_path in args.scenes:
            # If the scene path is inside SCENES_DIR, get the relative path
            try:
                rel_path = os.path.relpath(scene_path, SCENES_DIR)
            except ValueError:
                # Path is on different drive (Windows) or not relative
                rel_path = scene_path

            # Extract the top-level directory (images or animations)
            parts = rel_path.split(os.sep)
            if len(parts) > 0:
                scene_type = parts[0]
                if scene_type not in scene_groups:
                    print_warning(
                        f"Skipping unrecognized scene type: {scene_path} (not in 'images' or 'animations' directory)"
                    )
                    continue
            else:
                print_warning(f"Skipping invalid scene: {scene_path}")
                continue
            scene_list.append((scene_path, scene_type))
    else:
        # Find all scenes in images/ and animations/ directories
        for group in scene_groups:
            scene_dir = os.path.join(SCENES_DIR, group)
            scenes = find_scenes(scene_dir)
            for scene_path in scenes:
                scene_list.append((scene_path, group))

    # Process scenes
    try:
        for scene_path, scene_type in scene_list:
            # Skip non-image/animation scenes
            if scene_type not in ["images", "animations"]:
                print_warning(f"Skipping unknown scene type: {scene_type}")
                summary["unexpected_failures"].append(
                    f"Unsupported scene type: {scene_type} - {scene_path}"
                )
                continue

            # Normalize scene_type to singular
            scene_group = scene_type.rstrip("s")

            # Run test render task
            summary["total"] += 1
            test_result = run_render_task(
                scene_path,
                scene_group,
                "test",
                output_dir,
                skip_comparison,
                show_output,
                args.hide_progress_bars,
            )

            # Handle result
            test_result["scene"] = scene_path
            record_result(test_result, summary)

            # Conditionally run showcase render
            if args.command == "render":
                rel_scene_path = os.path.relpath(scene_path, SCENES_DIR)
                if rel_scene_path in SHOWCASE_SCENES:
                    summary["total"] += 1
                    showcase_result = run_render_task(
                        scene_path,
                        scene_group,
                        "showcase",
                        Path(SHOWCASE_DIR),
                        skip_comparison,
                        show_output,
                        args.hide_progress_bars,
                    )
                    showcase_result["scene"] = f"[SHOWCASE] {scene_path}"
                    record_result(showcase_result, summary)

        passed = print_summary(summary, args.command)

        # Clean up temporary files
        if not skip_comparison:
            shutil.rmtree(output_dir)

        if not passed:
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


def run_render_task(
    scene_path,
    scene_group,
    preset,
    output_dir,
    skip_comparison,
    show_output,
    hide_progress_bars=False,
):
    """Run a single render task and return result dictionary"""
    # Get configuration for the preset and scene type
    config = RENDER_PRESETS.get(preset)
    if not config:
        return {
            "type": "unexpected_failure",
            "message": f"Unknown render preset: {preset}",
        }

    type_config = config.get(scene_group)
    if not type_config:
        return {
            "type": "unexpected_failure",
            "message": f"Unknown scene group: {scene_group} for preset {preset}",
        }

    # Compute output file path
    scene_name = os.path.splitext(os.path.relpath(scene_path, SCENES_DIR))[0]
    output_rel_path = Path(output_dir) / f"{scene_name}{type_config['ext']}"
    output_dir_path = output_rel_path.parent if preset != "showcase" else output_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Convert string dictionary to list of flags [k, v] pairs
    def dict_to_flags(flag_dict):
        flags = []
        for k, v in flag_dict.items():
            flags.append(k)
            flags.append(v)
        return flags

    # Handle configuration using dictionaries to avoid duplicated flags
    applied_config = dict(type_config)  # Copy base config

    # Get and convert all flags from config
    general_flags = {}
    general_flags.update(config.get("general", {}))
    general_flags.update(type_config.get("general_flags", {}))
    render_flags = type_config.get("subcommand_flags", {})

    # Apply showcase-specific overrides if needed
    if preset == "showcase":
        rel_path = os.path.relpath(scene_path, SCENES_DIR)
        if rel_path in SHOWCASE_SCENES:
            scene_config = SHOWCASE_SCENES[rel_path] or {}
            # Apply general flags overrides
            if "general" in scene_config:
                general_flags.update(scene_config["general"])
            # Apply type-specific flags overrides
            type_overrides = scene_config.get(scene_group)
            if type_overrides and "subcommand_flags" in type_overrides:
                render_flags.update(type_overrides["subcommand_flags"])
            # Apply general type-specific overrides
            if type_overrides and "general_flags" in type_overrides:
                general_flags.update(type_overrides["general_flags"])

    # Build combined flags list
    general_flags = config.get("general", [])
    if "general" in applied_config:
        general_flags.extend(applied_config["general"])

    # Convert the flag dictionaries to lists
    general_flags_list = dict_to_flags(general_flags)
    render_flags_list = dict_to_flags(render_flags)

    # Add hide progress bar flag if requested
    if hide_progress_bars:
        general_flags_list.append("--hide-progress-bar")

    # Build render command in correct order:
    # renderer [general flags] --output-dir <output dir> <scene> subcommand [subcommand flags]
    cmd = [
        RENDERER,
        *general_flags_list,
        "--output-dir",
        str(output_dir_path),
        scene_path,
        *applied_config["commands"],
        *render_flags_list,
    ]

    print(format_color("\n" + "=" * 50, Color.MAGENTA))
    print(format_color(f"Rendering {preset}: {scene_path}", Color.BOLD))
    print(format_color(f"Command: {' '.join(cmd)}", Color.CYAN))
    print(format_color("=" * 50, Color.MAGENTA))

    start_time = time.time()

    output, interrupted, returncode = run_command_with_pty(cmd, show_output)

    elapsed = time.time() - start_time

    # On successful render & comparison required, compare against golden reference
    result = {
        "scene": scene_path,
        "elapsed": elapsed,
    }

    if interrupted:
        result.update(
            {
                "type": "render_failure",
                "message": "Renderer interrupted by user",
            }
        )
    elif returncode != 0:
        # Extract last 10 lines from bytes buffer
        lines = output.splitlines()
        last_10 = lines[-10:] if len(lines) > 10 else lines
        last_10_str = "\n".join(last_10)
        failure_msg = f"Renderer failed. Last 10 lines of output:\n{last_10_str}"
        result.update(
            {
                "type": "render_failure",
                "message": failure_msg,
            }
        )
    else:
        result.update(
            {
                "type": "passed",
                "message": f"Rendered in {elapsed:.2f}s",
            }
        )

        # For test preset and if we should compare, run the comparison
        if preset == "test" and not skip_comparison:
            golden_path = os.path.join(GOLDEN_DIR, f"{scene_name}{type_config['ext']}")
            if not os.path.exists(golden_path):
                # If golden reference is missing
                result.update(
                    {
                        "type": "missing_reference",
                        "message": "Missing golden reference",
                    }
                )
            else:
                # Compare the rendered file with the golden reference
                comparison_result = compare_output(
                    scene_group,
                    golden_path,
                    output_rel_path,
                    elapsed,
                )
                comparison_result["scene"] = scene_name
                # Update the result with comparison result
                result.update(comparison_result)

    print(format_color("-" * 50, Color.MAGENTA))
    return result


def compare_images(golden_path, test_path, elapsed_time=None):
    """Compare two images using ssimulacra2_rs"""
    try:
        cmd = ["ssimulacra2_rs", "image", golden_path, str(test_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return {
                "type": "regression",
                "message": f"Comparison failed (exit code {result.returncode})",
            }

        # Parse SSIM score from output
        output = result.stdout.strip()
        try:
            score = float(output.split()[-1])
        except (IndexError, ValueError):
            return {
                "type": "regression",
                "message": "Failed to parse SSIM score",
            }

        # Evaluate against threshold
        base_msg = f" in {elapsed_time:.2f}s" if elapsed_time is not None else ""
        if score >= SSIM_THRESHOLD:
            return {
                "type": "passed",
                "message": f"Image pass{base_msg} with SSIM: {score}",
            }
        else:
            return {
                "type": "regression",
                "message": f"Difference exceeds threshold (SSIM={score:.1f} < {SSIM_THRESHOLD})",
            }

    except FileNotFoundError:
        return {
            "type": "regression",
            "message": "Comparison tool missing",
        }


def compare_animations(golden_path, test_path, elapsed_time):
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
                }

            if not extract_video_frames(test_path, test_dir):
                return {
                    "type": "unexpected_failure",
                    "message": "Failed to extract test frames",
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
                }
            else:
                return {
                    "type": "regression",
                    "message": (
                        f"Difference in frame {worst_frame} exceeds threshold "
                        f"(SSIM={worst_score:.1f} < {SSIM_THRESHOLD})"
                    ),
                }

    except Exception as e:
        return {
            "type": "regression",
            "message": f"Animation comparison error: {str(e)}",
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


def print_summary(summary, command_type):
    """Print test summary with details"""

    plural_command_type = command_type + "s"

    print("\n" + format_color("=" * 50, Color.MAGENTA))
    print(format_color(f"{'Test Summary':^50}", Color.MAGENTA + Color.BOLD))
    print(format_color("=" * 50, Color.MAGENTA))

    total = summary["total"]
    # Define result type configuration once
    result_types = {
        "passed": {
            "color": Color.GREEN,
            "label": "Passed",
            "plural": "tests",
            "header": "Passes:",
        },
        "render_failure": {
            "color": Color.RED,
            "label": "Render fails",
            "plural": "fails",
            "header": "Render Failures:",
        },
        "regression": {
            "color": Color.RED,
            "label": "Regressions",
            "plural": "regressions",
            "header": "Regressions:",
        },
        "unexpected_failure": {
            "color": Color.RED,
            "label": "Unexpected fails",
            "plural": "fails",
            "header": "Unexpected failures:",
        },
        "missing_reference": {
            "color": Color.YELLOW,
            "label": "Missing refs",
            "plural": "refs",
            "header": "Missing References:",
        },
    }

    # Filter and count all result types in one pass
    results_by_type = {
        typ: [r for r in summary["results"] if r["type"] == typ] for typ in result_types
    }
    counts_by_type = {typ: len(results) for typ, results in results_by_type.items()}

    # Print counts
    print(f"Total tests:      {format_color(str(total), Color.CYAN)}")
    for typ, config in result_types.items():
        print(
            f"{config['label'] + ':':<17} {format_color(str(counts_by_type[typ]), config['color'])}"
        )

    # Print detailed breakdowns
    for typ, config in result_types.items():
        results = results_by_type[typ]
        if results:
            print("\n" + format_color(config["header"], config["color"] + Color.BOLD))
            for r in results:
                print(f"  • {r['scene']}: {r['message']}")
        elif typ == "passed":
            print(
                format_color(
                    f"No {plural_command_type} passed.", Color.YELLOW + Color.BOLD
                )
            )

    print(format_color("=" * 50, Color.MAGENTA))

    # Final status
    all_passed = counts_by_type["passed"] == total

    upper_case_command = plural_command_type.upper()
    if all_passed:
        final_msg = f"ALL {upper_case_command} PASSED"
        color = Color.GREEN
    else:
        final_msg = f"{upper_case_command} FAILED"
        color = Color.RED

    print(format_color(f"\n{final_msg}", color + Color.BOLD))
    print(format_color("=" * 50, Color.MAGENTA))

    return all_passed


if __name__ == "__main__":
    sys.exit(main())
