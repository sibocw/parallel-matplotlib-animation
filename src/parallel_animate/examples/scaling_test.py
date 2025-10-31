import time
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from parallel_animate.examples.very_complex_animation import VeryComplexAnimation


def run_scaling_test(
    num_frames: int, num_workers_list: list[int], output_dir: Path
) -> dict[str, dict]:
    """Run strong scaling test for different parallelization strategies."""
    # Generate parameters
    params = [{"phase": 2 * np.pi * i / num_frames} for i in range(num_frames)]

    # Run tests for all configurations
    results_all = {}
    for num_workers in num_workers_list:
        for reuse_figure_object in [True, False]:
            config_name = f"{num_workers}workers_reusefig{reuse_figure_object}"
            print(f"Running test: {config_name}...")
            output_file = output_dir / f"output_{config_name}.mp4"
            start_time = time.time()
            anim = VeryComplexAnimation()
            anim.make_video(
                output_file=output_file,
                param_by_frame=params,
                fps=30,
                num_workers=num_workers,
                disable_progress_bar=True,
                reuse_figure_object=reuse_figure_object,
            )
            elapsed_time = time.time() - start_time
            results_all[config_name] = {
                "num_workers": num_workers,
                "reuse_fig_obj": reuse_figure_object,
                "time_seconds": elapsed_time,
            }
            print(f"Test {config_name} completed in {elapsed_time:.2f} seconds")

    return results_all


def plot_scaling_results(results: dict[str, dict], output_path: Path) -> None:
    """Create plots showing scaling performance."""
    num_workers_list = list(set(result["num_workers"] for result in results.values()))

    fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
    baseline_time = results[f"1workers_reusefigFalse"]["time_seconds"]
    _max_speedup = len(num_workers_list)
    for reuse_figure_object in [True, False]:
        times = [
            results[f"{n}workers_reusefig{reuse_figure_object}"]["time_seconds"]
            for n in num_workers_list
        ]
        speedups = [baseline_time / t for t in times]
        _max_speedup = max(_max_speedup, *speedups)
        ax.plot(
            num_workers_list,
            speedups,
            marker="o",
            label="with cache" if reuse_figure_object else "without cache",
        )
    ax.plot(
        [1, max(num_workers_list)],
        [1, max(num_workers_list)],
        color="black",
        label="ideal scaling",
        zorder=0,
    )
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("# workers")
    ax.set_ylabel("speedup")
    ax.set_title("Strong scaling test")
    ax.set_aspect("equal")
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    output_dir = Path("example_output/scaling_test")
    output_dir.mkdir(exist_ok=True, parents=True)

    num_frames_to_draw = 320
    num_workers_to_test = [1, 2, 4, 8, 16]
    results = run_scaling_test(num_frames_to_draw, num_workers_to_test, output_dir)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    with open(output_dir / "results.json", "r") as f:
        results = json.load(f)
    plot_scaling_results(results, output_path=output_dir / "scaling_graph.png")
