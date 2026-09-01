"""Benchmark traversal and color-choice modes in ``coloringGame.py``.

Like ``test.py``, this script prepares Stanford SNAP datasets, records JSON
results, and can generate execution-time and color-count plots.  It also uses
SNAP to generate reproducible G(n,m) random graphs under ``out/generated``.
For every selected or generated graph it runs the same six
one-factor-at-a-time mode cases.  In every group, only the named mode is
varied and the other tested mode keeps its default.  The all-default
configuration intentionally appears once per group.

Mode meanings
-------------
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

"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from test import (
    DEFAULT_DATA_DIR,
    SNAP_HOME,
    count_conflicts,
    ensure_dir,
    get_dataset_registry,
    graph_stats,
    normalize_coloring,
    prepare_dataset,
    safe_file_stem,
)


DEFAULT_RESULT_PATH = Path("result") / "coloring_game_mode_results.json"
DEFAULT_PLOT_DIR = Path("result") / "gaming_mode_plots"
DEFAULT_GENERATED_GRAPH_DIR = Path("out") / "generated"
DEFAULT_DATASETS = ("ca-GrQc", "facebook_combined", "as20000102")
DEFAULT_GENERATED_GRAPH_COUNT = 10
DEFAULT_GENERATED_NODE_COUNT = 50
DEFAULT_GENERATED_EDGE_PROBABILITY = 0.1
DEFAULT_GENERATED_GRAPH_SEED = 42
DEFAULT_TRAVERSE_MODE = "forward"
DEFAULT_COLOR_CHOICE_MODE = "forward"

# snap.TRnd(0) is not reproducible, and the maximum signed-int seed can make
# GenRndGnm degenerate.  Normalize generated-graph seeds to this safe range.
MIN_SNAP_SEED = 1
MAX_SNAP_SEED = 2_147_483_646
SNAP_SEED_COUNT = MAX_SNAP_SEED - MIN_SNAP_SEED + 1


@dataclass(frozen=True)
class ModeCase:
    """One isolated experiment; the non-target tested mode stays default."""

    case_id: str
    changed_mode: str
    description: str
    traverse_mode: str = DEFAULT_TRAVERSE_MODE
    color_choice_mode: str = DEFAULT_COLOR_CHOICE_MODE


@dataclass(frozen=True)
class GeneratedGraphSpec:
    """Configuration for one reproducible SNAP G(n,m) graph."""

    graph_index: int
    node_count: int
    edge_probability: float
    seed: int

    @property
    def maximum_edge_count(self) -> int:
        return self.node_count * (self.node_count - 1) // 2

    @property
    def edge_count(self) -> int:
        return int(self.edge_probability * self.maximum_edge_count)

    @property
    def actual_density(self) -> float:
        if self.maximum_edge_count == 0:
            return 0.0
        return self.edge_count / self.maximum_edge_count

    @property
    def graph_id(self) -> str:
        probability_tag = format(self.edge_probability, ".12g").replace(
            ".", "p"
        )
        return (
            f"gaming_n{self.node_count}_p{probability_tag}_"
            f"run{self.graph_index:02d}_seed{self.seed}"
        )


MODE_CASES: Sequence[ModeCase] = (
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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def edge_probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("probability must be between 0 and 1")
    return parsed


def snap_seed(value: str) -> int:
    parsed = int(value)
    if not MIN_SNAP_SEED <= parsed <= MAX_SNAP_SEED:
        raise argparse.ArgumentTypeError(
            f"seed must be between {MIN_SNAP_SEED} and {MAX_SNAP_SEED}"
        )
    return parsed


def build_generated_graph_specs(args: argparse.Namespace) -> List[GeneratedGraphSpec]:
    specs: List[GeneratedGraphSpec] = []
    for offset in range(args.generated_graph_count):
        seed = (
            (args.generated_graph_seed - MIN_SNAP_SEED + offset)
            % SNAP_SEED_COUNT
        ) + MIN_SNAP_SEED
        specs.append(
            GeneratedGraphSpec(
                graph_index=offset + 1,
                node_count=args.generated_node_count,
                edge_probability=args.generated_edge_probability,
                seed=seed,
            )
        )
    return specs


def load_snap():
    """Import SNAP lazily so --help and --describe-only work without it."""

    try:
        return importlib.import_module("snap")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'snap' module is required to generate random graphs. Install "
            "requirements.txt or use the project's durham Conda interpreter."
        ) from exc


def generate_random_graph(
    snap_module,
    spec: GeneratedGraphSpec,
    graph_dir: Path,
) -> Tuple[Path, Dict[str, object]]:
    """Generate one seeded SNAP graph and save it with a gaming prefix."""

    ensure_dir(graph_dir)
    graph_path = graph_dir / f"{spec.graph_id}.txt"
    started_at = time.perf_counter()
    graph = snap_module.GenRndGnm(
        snap_module.TUNGraph,
        spec.node_count,
        spec.edge_count,
        False,
        snap_module.TRnd(spec.seed),
    )
    snap_module.SaveEdgeList(
        graph,
        str(graph_path),
        (
            "SNAP random undirected G(n,m) graph for testGaming; "
            f"requested_n={spec.node_count}, m={spec.edge_count}, "
            f"requested_p={spec.edge_probability}, seed={spec.seed}, "
            f"graph_index={spec.graph_index}"
        ),
    )
    generation_time = time.perf_counter() - started_at

    if graph.GetNodes() != spec.node_count or graph.GetEdges() != spec.edge_count:
        raise RuntimeError(
            f"SNAP generated {graph.GetNodes()} nodes and {graph.GetEdges()} edges; "
            f"expected {spec.node_count} nodes and {spec.edge_count} edges."
        )

    isolated_node_ids = sorted(
        node.GetId() for node in graph.Nodes() if node.GetDeg() == 0
    )
    if isolated_node_ids:
        with graph_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "# Isolated node IDs (restored through a temporary test input): "
                + " ".join(str(node_id) for node_id in isolated_node_ids)
                + "\n"
            )

    saved_stats = graph_stats(graph_path)
    if saved_stats["edges"] != spec.edge_count:
        raise RuntimeError(
            f"Saved graph contains {saved_stats['edges']} edges; "
            f"expected {spec.edge_count}."
        )
    if saved_stats["nodes"] + len(isolated_node_ids) != spec.node_count:
        raise RuntimeError(
            "Saved edge-list nodes plus isolated nodes do not match the "
            f"requested node count {spec.node_count}."
        )

    return graph_path, {
        "nodes": spec.node_count,
        "edge_list_nodes": saved_stats["nodes"],
        "isolated_nodes": len(isolated_node_ids),
        "isolated_node_ids": isolated_node_ids,
        "edges": saved_stats["edges"],
        "requested_edge_probability": spec.edge_probability,
        "actual_density": round(spec.actual_density, 12),
        "graph_seed": spec.seed,
        "graph_index": spec.graph_index,
        "generation_time_sec": round(generation_time, 6),
    }


@contextlib.contextmanager
def complete_generated_graph(
    graph_path: Path,
    isolated_node_ids: Sequence[int],
):
    """Yield a loader input that retains isolated nodes via temporary loops."""

    if not isolated_node_ids:
        yield graph_path.resolve()
        return

    with tempfile.TemporaryDirectory(prefix="gaming-graph-input-") as temp_dir:
        complete_path = Path(temp_dir) / graph_path.name
        shutil.copyfile(graph_path, complete_path)
        with complete_path.open("a", encoding="utf-8") as handle:
            for node_id in isolated_node_ids:
                handle.write(f"{node_id}\t{node_id}\n")
        yield complete_path


def validate_mode_cases(cases: Sequence[ModeCase] = MODE_CASES) -> None:
    """Guard the requested six-case, one-factor-at-a-time test design."""

    if len(cases) != 6:
        raise ValueError(f"Exactly 6 mode cases are required; got {len(cases)}.")

    expected_groups = {
        "traverse_mode": 3,
        "color_choice_mode": 3,
    }
    actual_groups = {
        group: sum(case.changed_mode == group for case in cases)
        for group in expected_groups
    }
    if actual_groups != expected_groups:
        raise ValueError(
            f"Each tested mode must have exactly 3 cases; got {actual_groups}."
        )

    for case in cases:
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
    print("  traverse_mode (default 'forward'):")
    print("    forward = ascending IDs; reverse = descending IDs; random = seeded shuffle.")
    print("  color_choice_mode (default 'forward'):")
    print("    forward/reverse/random control candidate-color order and payoff tie-breaking.")
    print("Test design: 3 traversal + 3 color-choice cases = 6 cases.")
    print(
        "Generated graph defaults: "
        f"{DEFAULT_GENERATED_GRAPH_COUNT} SNAP G(n,m) graphs, "
        f"n={DEFAULT_GENERATED_NODE_COUNT}, "
        f"p={DEFAULT_GENERATED_EDGE_PROBABILITY}, "
        f"saved under {DEFAULT_GENERATED_GRAPH_DIR} with prefix gaming."
    )


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
                    random_seed=random_seed,
                )
        finally:
            coloring_game_module.GRAPH_PATH = original_graph_path

    with _output_context(verbose):
        agent.load_graph(str(graph_path))

    # Temporary self-loops are loader-only markers for isolated vertices in
    # generated edge lists.  Remove the loops but keep the node objects.
    self_loop_nodes = [
        edge.GetSrcNId()
        for edge in agent.graph.Edges()
        if edge.GetSrcNId() == edge.GetDstNId()
    ]
    for node_id in self_loop_nodes:
        agent.graph.DelEdge(node_id, node_id)
    if self_loop_nodes:
        agent._rebuild_state()

    # Configure traversal last: reset_stepper() then stores the first seeded
    # traversal (including the first random shuffle) in _step_sequence.
    agent.set_color_choice_mode(case.color_choice_mode)
    agent.set_traverse_mode(case.traverse_mode)
    return agent


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
    # reset_stepper() has already saved the first traversal in this sequence.
    traverse_sequence = list(agent._step_sequence)
    color_choice_sequence = agent.get_color_sequence()

    expected_traverse = _expected_sequence(
        node_ids, case.traverse_mode, random_seed
    )
    expected_colors = _expected_sequence(
        list(agent.available_colors), case.color_choice_mode, random_seed
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
        "random_seed": random_seed,
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


def run_mode_cases_for_graph(
    graph_path: Path,
    graph_report: Dict[str, object],
    args: argparse.Namespace,
    all_results: List[Dict[str, object]],
    random_seed: int,
    expected_node_count: Optional[int] = None,
) -> None:
    """Run the fixed six mode cases and append them to one graph report."""

    print(
        f"Graph {graph_report['dataset']}: source={graph_report['source_type']}, "
        f"nodes={graph_report['nodes']}, edges={graph_report['edges']}, "
        f"graph={graph_report['graph_path']}"
    )
    for index, case in enumerate(MODE_CASES, start=1):
        print(
            f"  [{index}/{len(MODE_CASES)}] {case.case_id}: "
            f"traverse={case.traverse_mode}, "
            f"color_choice={case.color_choice_mode}"
        )
        try:
            result = run_mode_case(
                case=case,
                graph_path=graph_path,
                random_seed=random_seed,
                verbose=args.verbose,
            )
            if expected_node_count is not None:
                expected_nodes = set(range(expected_node_count))
                colored_nodes = {int(node_id) for node_id in result["coloring"]}
                if colored_nodes != expected_nodes:
                    missing = sorted(expected_nodes - colored_nodes)
                    unexpected = sorted(colored_nodes - expected_nodes)
                    raise AssertionError(
                        "Coloring does not cover the generated node set; "
                        f"missing={missing}, unexpected={unexpected}."
                    )
        except Exception as exc:
            result = {
                **asdict(case),
                "status": "error",
                "random_seed": random_seed,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"    status=error, error={result['error']}")
        else:
            print(
                f"    status=ok, time={result['execution_time_sec']}s, "
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

        graph_report["results"].append(result)
        all_results.append(result)


def _case_plot_label(result: Dict[str, object]) -> str:
    changed_mode = str(result["changed_mode"])
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
            "Run 6 one-factor-at-a-time coloring-game mode tests on SNAP "
            "datasets and generated random graphs."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=sorted(dataset_registry),
        help=(
            "Named datasets to run in addition to generated graphs "
            "(default: the same 3 SNAP datasets as test.py)."
        ),
    )
    parser.add_argument(
        "--generated-only",
        action="store_true",
        help="Skip named SNAP datasets and test only generated random graphs.",
    )
    generated_count_group = parser.add_mutually_exclusive_group()
    generated_count_group.add_argument(
        "--generated-graph-count",
        "--generated-count",
        dest="generated_graph_count",
        type=non_negative_int,
        default=DEFAULT_GENERATED_GRAPH_COUNT,
        help="Number of random graphs to generate and test (default: 10).",
    )
    generated_count_group.add_argument(
        "--no-generated-graphs",
        action="store_const",
        const=0,
        dest="generated_graph_count",
        help="Disable the generated-random-graph data source.",
    )
    parser.add_argument(
        "--generated-node-count",
        type=positive_int,
        default=DEFAULT_GENERATED_NODE_COUNT,
        help="Number of nodes in each generated graph (default: 50).",
    )
    parser.add_argument(
        "--generated-edge-probability",
        type=edge_probability,
        default=DEFAULT_GENERATED_EDGE_PROBABILITY,
        help="Density converted to m=floor(p*n*(n-1)/2) (default: 0.1).",
    )
    parser.add_argument(
        "--generated-graph-seed",
        type=snap_seed,
        default=DEFAULT_GENERATED_GRAPH_SEED,
        help="Base SNAP seed for generated graphs (default: 42).",
    )
    parser.add_argument(
        "--generated-graph-dir",
        "--generated-dir",
        dest="generated_graph_dir",
        type=Path,
        default=DEFAULT_GENERATED_GRAPH_DIR,
        help="Generated graph directory (default: out/generated).",
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
        help="Print mode meanings and the six cases without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_mode_cases()
    print_mode_explanation()

    if args.describe_only:
        for index, case in enumerate(MODE_CASES, start=1):
            print(
                f"  {index}. {case.case_id}: traverse={case.traverse_mode}, "
                f"color_choice={case.color_choice_mode}"
            )
        return

    if args.generated_only and args.generated_graph_count == 0:
        raise SystemExit(
            "No graph source selected: --generated-only cannot be combined "
            "with zero generated graphs."
        )

    dataset_registry = get_dataset_registry()
    max_nodes = None if args.max_nodes <= 0 else args.max_nodes
    selected_sources: List[str] = []
    if args.generated_graph_count > 0:
        selected_sources.append("snap_generated")
    if not args.generated_only:
        selected_sources.append("snap_dataset")
    report: Dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": ", ".join(selected_sources),
        "selected_sources": selected_sources,
        "data_sources": {
            "snap_generated": {
                "enabled": args.generated_graph_count > 0,
                "description": "Synthetic random G(n,m) graphs generated by SNAP.",
            },
            "snap_dataset": {
                "enabled": not args.generated_only,
                "description": "Stanford Large Network Dataset Collection.",
                "url": SNAP_HOME,
            },
        },
        "data_source_url": None if args.generated_only else SNAP_HOME,
        "max_nodes": max_nodes,
        "named_dataset_max_nodes": max_nodes,
        "random_seed": args.random_seed,
        "mode_random_seed": args.random_seed,
        "generated_graphs": {
            "enabled": args.generated_graph_count > 0,
            "count": args.generated_graph_count,
            "graph_dir": str(args.generated_graph_dir),
            "file_prefix": "gaming",
            "graph_model": "G(n,m)",
            "generator": (
                "snap.GenRndGnm(snap.TUNGraph, n, m, False, snap.TRnd(seed))"
            ),
            "node_count": args.generated_node_count,
            "edge_probability": args.generated_edge_probability,
            "base_graph_seed": args.generated_graph_seed,
        },
        "defaults": {
            "traverse_mode": DEFAULT_TRAVERSE_MODE,
            "color_choice_mode": DEFAULT_COLOR_CHOICE_MODE,
        },
        "test_design": (
            "One factor at a time: 3 traversal + 3 color-choice cases."
        ),
        "case_count_per_dataset": len(MODE_CASES),
        "case_count_per_graph": len(MODE_CASES),
        "datasets": [],
    }

    ensure_dir(args.result_path.parent)
    all_results: List[Dict[str, object]] = []
    generated_graph_files: List[str] = []

    generated_specs = build_generated_graph_specs(args)
    if generated_specs:
        try:
            snap_module = load_snap()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

        ensure_dir(args.generated_graph_dir)
        for spec in generated_specs:
            graph_path, stats = generate_random_graph(
                snap_module=snap_module,
                spec=spec,
                graph_dir=args.generated_graph_dir,
            )
            generated_graph_files.append(str(graph_path))
            graph_report: Dict[str, object] = {
                "dataset": spec.graph_id,
                "source_type": "snap_generated",
                "source_url": None,
                "description": "Seeded SNAP random undirected G(n,m) graph.",
                "graph_path": str(graph_path),
                "graph_model": "G(n,m)",
                "generator": (
                    "snap.GenRndGnm(snap.TUNGraph, n, m, False, "
                    "snap.TRnd(seed))"
                ),
                **stats,
                "results": [],
            }
            if stats["isolated_nodes"]:
                print(
                    f"Graph {spec.graph_id} contains "
                    f"{stats['isolated_nodes']} isolated node(s); preserving "
                    "them through a temporary loader-only input."
                )

            with complete_generated_graph(
                graph_path,
                stats["isolated_node_ids"],
            ) as complete_graph_path:
                run_mode_cases_for_graph(
                    graph_path=complete_graph_path,
                    graph_report=graph_report,
                    args=args,
                    all_results=all_results,
                    random_seed=args.random_seed,
                    expected_node_count=spec.node_count,
                )
            report["datasets"].append(graph_report)

    if not args.generated_only:
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
                "source_type": "snap_dataset",
                "description": spec.description,
                "source_url": spec.source_url,
                "graph_path": str(graph_path),
                "nodes": stats["nodes"],
                "edges": stats["edges"],
                "results": [],
            }
            run_mode_cases_for_graph(
                graph_path=graph_path,
                graph_report=dataset_report,
                args=args,
                all_results=all_results,
                random_seed=args.random_seed,
            )
            report["datasets"].append(dataset_report)

    report["generated_graphs"]["files"] = generated_graph_files
    report["total_case_runs"] = len(all_results)
    with args.result_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Full mode benchmark results saved to {args.result_path}")
    if generated_graph_files:
        print(
            f"Generated {len(generated_graph_files)} graph file(s) under "
            f"{args.generated_graph_dir} with prefix gaming."
        )

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
