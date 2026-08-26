"""Benchmark the three mode families in ``coloringGame.py``.

Like ``test.py``, this script prepares Stanford SNAP datasets, records JSON
results, and can generate execution-time and color-count plots.  For every
selected dataset it runs the same nine one-factor-at-a-time mode cases.  In
every group, only the named mode is varied and the other two modes keep their
defaults.  The all-default configuration intentionally appears once per
group.

Mode meanings
-------------
``node_mode`` (default: 0)
    0 selects the node with the smallest ID, 1 selects the node with the
    largest ID, and 2 selects a node pseudo-randomly using ``random_seed``.
    The current implementation treats every value other than 0 and 1 as the
    random branch; this test uses 2 as its canonical name.

``traverse_mode`` (default: ``"forward"``)
    Controls the order in which nodes are visited in each improvement round:
    ascending node ID (forward), descending node ID (reverse), or a seeded
    shuffled order (random).

``color_choice_mode`` (default: ``"forward"``)
    Controls the order in which candidate color IDs are examined: ascending
    (forward), descending (reverse), or seeded shuffled order (random).  Since
    the algorithm scans every color and only replaces the current best on a
    strict payoff improvement, this order breaks ties between equally good
    colors; it is not a first-improvement policy.

Important implementation detail: ``node_mode`` currently affects only
``get_one_node()``.  ``move_to_nash_equilibrium()`` visits nodes through
``traverse_mode`` and does not call ``get_one_node()``.  Therefore the three
node-mode cases validate and record the selected node, but their final
colorings can be identical.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from test import (
    DEFAULT_DATA_DIR,
    SNAP_HOME,
    count_conflicts,
    ensure_dir,
    get_dataset_registry,
    normalize_coloring,
    prepare_dataset,
    safe_file_stem,
)


DEFAULT_RESULT_PATH = Path("result") / "coloring_game_mode_results.json"
DEFAULT_PLOT_DIR = Path("result") / "gaming_mode_plots"
DEFAULT_DATASETS = ("ca-GrQc", "facebook_combined", "as20000102")
DEFAULT_NODE_MODE = 0
DEFAULT_TRAVERSE_MODE = "forward"
DEFAULT_COLOR_CHOICE_MODE = "forward"


@dataclass(frozen=True)
class ModeCase:
    """One isolated mode experiment; the two non-target modes stay default."""

    case_id: str
    changed_mode: str
    description: str
    node_mode: int = DEFAULT_NODE_MODE
    traverse_mode: str = DEFAULT_TRAVERSE_MODE
    color_choice_mode: str = DEFAULT_COLOR_CHOICE_MODE


MODE_CASES: Sequence[ModeCase] = (
    ModeCase(
        case_id="node_mode_0",
        changed_mode="node_mode",
        node_mode=0,
        description="Select the node with the smallest node ID.",
    ),
    ModeCase(
        case_id="node_mode_1",
        changed_mode="node_mode",
        node_mode=1,
        description="Select the node with the largest node ID.",
    ),
    ModeCase(
        case_id="node_mode_2",
        changed_mode="node_mode",
        node_mode=2,
        description="Select one node pseudo-randomly using random_seed.",
    ),
    ModeCase(
        case_id="traverse_mode_forward",
        changed_mode="traverse_mode",
        traverse_mode="forward",
        description="Visit nodes in ascending node-ID order in every round.",
    ),
    ModeCase(
        case_id="traverse_mode_reverse",
        changed_mode="traverse_mode",
        traverse_mode="reverse",
        description="Visit nodes in descending node-ID order in every round.",
    ),
    ModeCase(
        case_id="traverse_mode_random",
        changed_mode="traverse_mode",
        traverse_mode="random",
        description="Visit nodes in a seeded shuffled order in every round.",
    ),
    ModeCase(
        case_id="color_choice_mode_forward",
        changed_mode="color_choice_mode",
        color_choice_mode="forward",
        description="Examine candidate color IDs in ascending order.",
    ),
    ModeCase(
        case_id="color_choice_mode_reverse",
        changed_mode="color_choice_mode",
        color_choice_mode="reverse",
        description="Examine candidate color IDs in descending order.",
    ),
    ModeCase(
        case_id="color_choice_mode_random",
        changed_mode="color_choice_mode",
        color_choice_mode="random",
        description="Examine candidate colors in a seeded shuffled order.",
    ),
)


def validate_mode_cases(cases: Sequence[ModeCase] = MODE_CASES) -> None:
    """Guard the requested nine-case, one-factor-at-a-time test design."""

    if len(cases) != 9:
        raise ValueError(f"Exactly 9 mode cases are required; got {len(cases)}.")

    expected_groups = {
        "node_mode": 3,
        "traverse_mode": 3,
        "color_choice_mode": 3,
    }
    actual_groups = {
        group: sum(case.changed_mode == group for case in cases)
        for group in expected_groups
    }
    if actual_groups != expected_groups:
        raise ValueError(
            f"Each mode must have exactly 3 cases; got {actual_groups}."
        )

    for case in cases:
        if case.changed_mode != "node_mode" and case.node_mode != DEFAULT_NODE_MODE:
            raise ValueError(f"{case.case_id}: node_mode must keep its default.")
        if (
            case.changed_mode != "traverse_mode"
            and case.traverse_mode != DEFAULT_TRAVERSE_MODE
        ):
            raise ValueError(f"{case.case_id}: traverse_mode must keep its default.")
        if (
            case.changed_mode != "color_choice_mode"
            and case.color_choice_mode != DEFAULT_COLOR_CHOICE_MODE
        ):
            raise ValueError(
                f"{case.case_id}: color_choice_mode must keep its default."
            )


def print_mode_explanation() -> None:
    print("Mode meanings:")
    print("  node_mode (default 0):")
    print("    0 = smallest node ID; 1 = largest node ID; 2 = seeded random node.")
    print("    It currently controls get_one_node(), not the equilibrium traversal.")
    print("  traverse_mode (default 'forward'):")
    print("    forward = ascending IDs; reverse = descending IDs; random = seeded shuffle.")
    print("  color_choice_mode (default 'forward'):")
    print("    forward/reverse/random control candidate-color order and payoff tie-breaking.")
    print("Test design: 3 node + 3 traversal + 3 color-choice cases = 9 cases.")


def _output_context(verbose: bool):
    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def _new_agent(case: ModeCase, graph_path: Path, random_seed: int, verbose: bool):
    """Create an agent without overwriting coloringGame.GRAPH_PATH."""

    import coloringGame as coloring_game_module

    with tempfile.TemporaryDirectory(prefix="coloring-game-mode-") as temp_dir:
        original_graph_path = coloring_game_module.GRAPH_PATH
        coloring_game_module.GRAPH_PATH = str(Path(temp_dir) / "generated_graph.txt")
        try:
            with _output_context(verbose):
                agent = coloring_game_module.coloring_game(
                    node_mode=case.node_mode,
                    random_seed=random_seed,
                )
        finally:
            coloring_game_module.GRAPH_PATH = original_graph_path

    with _output_context(verbose):
        agent.load_graph(str(graph_path))

    # Configure traversal last: reset_stepper() then stores the first seeded
    # traversal (including the first random shuffle) in _step_sequence.
    agent.set_node_mode(case.node_mode, random_seed=random_seed)
    agent.set_color_choice_mode(case.color_choice_mode)
    agent.set_traverse_mode(case.traverse_mode)
    return agent


def _expected_node(node_ids: List[int], node_mode: int, random_seed: int) -> int:
    if node_mode == 0:
        return min(node_ids)
    if node_mode == 1:
        return max(node_ids)
    return random.Random(random_seed).choice(node_ids)


def _expected_sequence(values: List[int], mode: str, random_seed: int) -> List[int]:
    sequence = sorted(values)
    if mode == "forward":
        return sequence
    if mode == "reverse":
        return list(reversed(sequence))
    if mode == "random":
        random.Random(random_seed).shuffle(sequence)
        return sequence
    raise ValueError(f"Unsupported sequence mode: {mode}")


def run_mode_case(
    case: ModeCase,
    graph_path: Path,
    random_seed: int,
    verbose: bool,
) -> Dict[str, object]:
    total_start = time.perf_counter()
    agent = _new_agent(case, graph_path, random_seed, verbose)

    node_ids = list(agent.node_ids)
    selected_node = agent.get_one_node().GetId()
    # reset_stepper() has already saved the first traversal in this sequence.
    traverse_sequence = list(agent._step_sequence)
    color_choice_sequence = agent.get_color_sequence()

    expected_node = _expected_node(node_ids, case.node_mode, random_seed)
    expected_traverse = _expected_sequence(
        node_ids, case.traverse_mode, random_seed
    )
    expected_colors = _expected_sequence(
        list(agent.available_colors), case.color_choice_mode, random_seed
    )

    if selected_node != expected_node:
        raise AssertionError(
            f"{case.case_id}: selected node {selected_node}, expected {expected_node}."
        )
    if traverse_sequence != expected_traverse:
        raise AssertionError(
            f"{case.case_id}: traversal {traverse_sequence}, "
            f"expected {expected_traverse}."
        )
    if color_choice_sequence != expected_colors:
        raise AssertionError(
            f"{case.case_id}: color sequence {color_choice_sequence}, "
            f"expected {expected_colors}."
        )

    solve_start = time.perf_counter()
    with _output_context(verbose):
        agent.move_to_nash_equilibrium()
    execution_time = time.perf_counter() - solve_start

    coloring = normalize_coloring(agent.node_color)
    conflicts = count_conflicts(graph_path, coloring)
    valid_coloring = agent.is_valid_coloring() and conflicts == 0
    nash_equilibrium = agent.is_nash_equilibrium()

    if set(coloring) != set(node_ids):
        raise AssertionError(f"{case.case_id}: coloring does not cover every node.")
    if not valid_coloring:
        raise AssertionError(
            f"{case.case_id}: final coloring has {conflicts} conflicting edges."
        )
    if not nash_equilibrium:
        raise AssertionError(f"{case.case_id}: Nash equilibrium was not reached.")

    return {
        **asdict(case),
        "status": "ok",
        "node_mode": case.node_mode,
        "traverse_mode": case.traverse_mode,
        "color_choice_mode": case.color_choice_mode,
        "selected_node": selected_node,
        "traverse_sequence": traverse_sequence,
        "color_choice_sequence": color_choice_sequence,
        "execution_time_sec": round(execution_time, 6),
        "total_time_sec": round(time.perf_counter() - total_start, 6),
        "color_count": len(agent.used_colors()),
        "used_colors": agent.used_colors(),
        "valid_coloring": valid_coloring,
        "conflicting_edges": conflicts,
        "nash_equilibrium": nash_equilibrium,
        "total_payoff": agent.get_total_payoff(),
        "coloring": coloring,
    }


def _case_plot_label(result: Dict[str, object]) -> str:
    changed_mode = str(result["changed_mode"])
    if changed_mode == "node_mode":
        return f"node:{result['node_mode']}"
    if changed_mode == "traverse_mode":
        return f"traverse:{result['traverse_mode']}"
    return f"color:{result['color_choice_mode']}"


def plot_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
    value_format: str,
) -> None:
    """Draw a mode-comparison chart in the same style as test.py."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    fig_width = max(10.0, 1.35 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    bars = ax.bar(labels, values, color="#2f6f8f")

    ax.set_title(title)
    ax.set_xlabel("Mode test case")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=30)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_mode_benchmark_images(
    report: Dict[str, object], plot_dir: Path
) -> List[Path]:
    """Create the time and color-count charts for every selected dataset."""

    generated: List[Path] = []
    for dataset_report in report["datasets"]:
        dataset_name = str(dataset_report["dataset"])
        dataset_stem = safe_file_stem(dataset_name)
        successful_results = [
            result
            for result in dataset_report["results"]
            if result.get("status") == "ok"
        ]

        time_items = [
            (_case_plot_label(result), float(result["execution_time_sec"]))
            for result in successful_results
            if isinstance(result.get("execution_time_sec"), (int, float))
        ]
        if time_items:
            labels, values = zip(*time_items)
            output_path = plot_dir / f"{dataset_stem}_gaming_mode_time.png"
            plot_bar_chart(
                labels=list(labels),
                values=list(values),
                title=f"{dataset_name} Coloring-game Mode Execution Time",
                ylabel="Time (seconds)",
                output_path=output_path,
                value_format="{:.4f}",
            )
            generated.append(output_path)

        color_items = [
            (_case_plot_label(result), float(result["color_count"]))
            for result in successful_results
            if isinstance(result.get("color_count"), (int, float))
        ]
        if color_items:
            labels, values = zip(*color_items)
            output_path = plot_dir / f"{dataset_stem}_gaming_mode_color_count.png"
            plot_bar_chart(
                labels=list(labels),
                values=list(values),
                title=f"{dataset_name} Coloring-game Mode Color Count",
                ylabel="Color count",
                output_path=output_path,
                value_format="{:.0f}",
            )
            generated.append(output_path)

    return generated


def parse_args() -> argparse.Namespace:
    dataset_registry = get_dataset_registry()
    parser = argparse.ArgumentParser(
        description=(
            "Run 9 one-factor-at-a-time coloring-game mode tests on SNAP datasets."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=sorted(dataset_registry),
        help="Dataset names to run, matching test.py.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1000,
        help="Use the first N dataset nodes; use 0 for the full graph.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-coloring", action="store_true")
    parser.add_argument("--coloring-print-limit", type=int, default=200)
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="Print mode meanings and the nine cases without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_mode_cases()
    print_mode_explanation()

    if args.describe_only:
        for index, case in enumerate(MODE_CASES, start=1):
            print(
                f"  {index}. {case.case_id}: node={case.node_mode}, "
                f"traverse={case.traverse_mode}, "
                f"color_choice={case.color_choice_mode}"
            )
        return

    dataset_registry = get_dataset_registry()
    max_nodes = None if args.max_nodes <= 0 else args.max_nodes
    report: Dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": "Stanford Large Network Dataset Collection",
        "data_source_url": SNAP_HOME,
        "max_nodes": max_nodes,
        "random_seed": args.random_seed,
        "defaults": {
            "node_mode": DEFAULT_NODE_MODE,
            "traverse_mode": DEFAULT_TRAVERSE_MODE,
            "color_choice_mode": DEFAULT_COLOR_CHOICE_MODE,
        },
        "test_design": (
            "One factor at a time: 3 node + 3 traversal + "
            "3 color-choice cases."
        ),
        "node_mode_scope": (
            "Currently affects get_one_node() only; the equilibrium loop is "
            "controlled by traverse_mode."
        ),
        "case_count_per_dataset": len(MODE_CASES),
        "datasets": [],
    }

    ensure_dir(args.result_path.parent)
    all_results: List[Dict[str, object]] = []

    for dataset_name in args.datasets:
        spec = dataset_registry[dataset_name]
        graph_path, stats = prepare_dataset(
            spec,
            data_dir=args.data_dir,
            max_nodes=max_nodes,
            force_download=args.force_download,
            force_rebuild=args.force_rebuild,
        )
        dataset_report: Dict[str, object] = {
            "dataset": spec.name,
            "description": spec.description,
            "source_url": spec.source_url,
            "graph_path": str(graph_path),
            "nodes": stats["nodes"],
            "edges": stats["edges"],
            "results": [],
        }

        print(
            f"Dataset {spec.name}: nodes={stats['nodes']}, "
            f"edges={stats['edges']}, graph={graph_path}"
        )
        for index, case in enumerate(MODE_CASES, start=1):
            print(
                f"  [{index}/9] {case.case_id}: node={case.node_mode}, "
                f"traverse={case.traverse_mode}, "
                f"color_choice={case.color_choice_mode}"
            )
            try:
                result = run_mode_case(
                    case=case,
                    graph_path=graph_path,
                    random_seed=args.random_seed,
                    verbose=args.verbose,
                )
            except Exception as exc:
                result = {
                    **asdict(case),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"    status=error, error={result['error']}")
            else:
                print(
                    f"    status=ok, time={result['execution_time_sec']}s, "
                    f"selected_node={result['selected_node']}, "
                    f"colors={result['color_count']}, "
                    f"valid={result['valid_coloring']}, "
                    f"nash={result['nash_equilibrium']}"
                )
                if args.print_coloring:
                    coloring = result["coloring"]
                    if len(coloring) <= args.coloring_print_limit:
                        print(f"    coloring={coloring}")
                    else:
                        print(
                            f"    coloring has {len(coloring)} vertices; "
                            "the full mapping is saved in the JSON result file."
                        )

            dataset_report["results"].append(result)
            all_results.append(result)

        report["datasets"].append(dataset_report)

    report["total_case_runs"] = len(all_results)
    with args.result_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Full mode benchmark results saved to {args.result_path}")

    if not args.no_plots:
        generated_plots = plot_mode_benchmark_images(report, args.plot_dir)
        for plot_path in generated_plots:
            print(f"Plot saved to {plot_path}")

    failed = [result for result in all_results if result["status"] != "ok"]
    if failed:
        raise SystemExit(
            f"{len(failed)} of {len(all_results)} mode test runs failed."
        )


if __name__ == "__main__":
    main()
