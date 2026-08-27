"""Benchmark graph-coloring algorithms on SNAP random undirected graphs.

This script follows the benchmark flow in ``test.py`` but replaces downloaded
datasets with reproducible SNAP G(n, m) graphs.  The clean generated edge
lists are saved under ``out/generated`` by default.  ColoringGame participates
with its class defaults only; this script exposes no ColoringGame mode options.

Example::

    python testRandom.py --graph-count 10 --edge-probabilities 0.1
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import random
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from test import (
    AlgorithmSpec,
    RunContext,
    count_conflicts,
    ensure_dir,
    get_algorithm_registry,
    graph_stats,
    normalize_coloring,
    plot_benchmark_images,
    print_algorithm_result,
    quiet_call,
    run_algorithm,
)


DEFAULT_GRAPH_DIR = Path("out") / "generated"
DEFAULT_RESULT_PATH = Path("result") / "random_coloring_benchmark_results.json"
DEFAULT_PLOT_DIR = Path("result") / "random_plots"
DEFAULT_GRAPH_COUNT = 10
DEFAULT_EDGE_PROBABILITIES = (0.1,)
MIN_RANDOM_NODES = 1
MAX_RANDOM_NODES = 200

# snap.TRnd(0) is not reproducible and the largest signed-int seed can make
# GenRndGnm degenerate.  Keep graph seeds inside the tested safe interval.
MIN_SNAP_SEED = 1
MAX_SNAP_SEED = 2_147_483_646
SNAP_SEED_COUNT = MAX_SNAP_SEED - MIN_SNAP_SEED + 1


@dataclass(frozen=True)
class RandomGraphSpec:
    node_count: int
    edge_probability: float
    graph_number: int
    seed: int

    @property
    def maximum_edge_count(self) -> int:
        return self.node_count * (self.node_count - 1) // 2

    @property
    def edge_count(self) -> int:
        # This is the same p-to-m conversion used by the project algorithms.
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
            f"random_g{self.graph_number:03d}_n{self.node_count}_"
            f"p{probability_tag}_seed{self.seed}"
        )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
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


def build_graph_specs(args: argparse.Namespace) -> List[RandomGraphSpec]:
    specs: List[RandomGraphSpec] = []
    node_rng = random.Random(args.random_seed)
    probabilities = [float(value) for value in args.edge_probabilities]

    for graph_index in range(args.graph_count):
        graph_number = graph_index + 1
        node_count = node_rng.randint(MIN_RANDOM_NODES, MAX_RANDOM_NODES)
        probability = probabilities[graph_index % len(probabilities)]
        seed = (
            (args.random_seed - MIN_SNAP_SEED + graph_index)
            % SNAP_SEED_COUNT
        ) + MIN_SNAP_SEED
        specs.append(
            RandomGraphSpec(
                node_count=node_count,
                edge_probability=probability,
                graph_number=graph_number,
                seed=seed,
            )
        )

    return specs


def load_snap():
    """Import SNAP lazily so ``--help`` works without SNAP installed."""

    try:
        return importlib.import_module("snap")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'snap' module is required. Install requirements.txt or run "
            "this script with the project's durham Conda interpreter."
        ) from exc


def generate_random_graph(
    snap_module,
    spec: RandomGraphSpec,
    graph_dir: Path,
) -> Tuple[Path, Dict[str, object]]:
    """Generate one seeded simple undirected graph and save its edge list."""

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
            "SNAP random undirected G(n,m) graph; "
            f"requested_n={spec.node_count}, m={spec.edge_count}, "
            f"requested_p={spec.edge_probability}, seed={spec.seed}, "
            f"graph_number={spec.graph_number}"
        ),
    )
    generation_time = time.perf_counter() - started_at

    if graph.GetNodes() != spec.node_count or graph.GetEdges() != spec.edge_count:
        raise RuntimeError(
            f"SNAP generated {graph.GetNodes()} nodes and {graph.GetEdges()} edges; "
            f"expected {spec.node_count} nodes and {spec.edge_count} edges."
        )

    saved_stats = graph_stats(graph_path)
    if saved_stats["edges"] != spec.edge_count:
        raise RuntimeError(
            f"Saved graph contains {saved_stats['edges']} edges; "
            f"expected {spec.edge_count}."
        )

    return graph_path, {
        "requested_nodes": spec.node_count,
        "nodes": saved_stats["nodes"],
        "edges": saved_stats["edges"],
        "requested_edge_probability": spec.edge_probability,
        "actual_density": round(spec.actual_density, 12),
        "graph_seed": spec.seed,
        "graph_number": spec.graph_number,
        "generation_time_sec": round(generation_time, 6),
    }


ALGORITHM_MODULES = {
    "coloring_game": "coloringGame",
    "ant_rlf": "ant_rlf",
    "tabucol": "tabucol",
    "dsatur": "dsatur",
}


def run_default_coloring_game(ctx: RunContext) -> Dict[str, object]:
    """Run ColoringGame with only its class defaults.

    No constructor argument or mode setter is used.  In particular, the
    algorithm always keeps ``node_mode=0``, forward traversal, forward color
    choice, and ``random_seed=42`` even when ``--random-seed`` changes the
    generated graph or the other randomized algorithms.
    """

    from coloringGame import coloring_game

    def execute():
        agent = coloring_game()
        agent.load_graph(str(ctx.graph_path))
        agent.move_to_nash_equilibrium()
        return agent

    agent = quiet_call(execute, ctx.args.verbose)
    coloring = normalize_coloring(agent.node_color)
    conflicts = count_conflicts(ctx.graph_path, coloring)
    return {
        "status": "ok",
        "color_count": len(set(coloring.values())),
        "valid_coloring": agent.is_valid_coloring() and conflicts == 0,
        "conflicting_edges": conflicts,
        "coloring": coloring,
        "parameters": {
            "node_mode": agent.node_mode,
            "traverse_mode": agent.traverse_mode,
            "color_choice_mode": agent.color_choice_mode,
            "random_seed": agent.random_seed,
        },
    }


def get_random_algorithm_registry() -> Dict[str, AlgorithmSpec]:
    """Return test.py algorithms plus the defaults-only ColoringGame case."""

    return {
        "coloring_game": AlgorithmSpec(
            "coloring_game",
            run_default_coloring_game,
        ),
        **get_algorithm_registry(),
    }


@contextlib.contextmanager
def temporary_constructor_graph(algorithm_name: str):
    """Prevent algorithm constructors from overwriting existing out files."""

    module_name = ALGORITHM_MODULES.get(algorithm_name)
    if module_name is None:
        yield
        return

    try:
        module = importlib.import_module(module_name)
        if algorithm_name == "tabucol":
            # run_tabucol imports this dependency inside the timed function.
            importlib.import_module("dsatur")
    except Exception:
        # run_algorithm will capture and report the same import error.
        yield
        return

    original_path = module.GRAPH_PATH
    with tempfile.TemporaryDirectory(prefix=f"{algorithm_name}-benchmark-") as temp_dir:
        module.GRAPH_PATH = str(Path(temp_dir) / "constructor_graph.txt")
        try:
            yield
        finally:
            module.GRAPH_PATH = original_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    algorithm_registry = get_random_algorithm_registry()
    parser = argparse.ArgumentParser(
        description=(
            "Generate seeded SNAP random undirected graphs and benchmark the "
            "project's graph-coloring algorithms."
        )
    )
    parser.add_argument(
        "--graph-count",
        type=positive_int,
        default=DEFAULT_GRAPH_COUNT,
        help=(
            "Total number of random graphs to generate; each graph gets a "
            "seeded random node count from 1 through 1000 (default: 3)."
        ),
    )
    parser.add_argument(
        "--edge-probabilities",
        nargs="+",
        type=edge_probability,
        default=list(DEFAULT_EDGE_PROBABILITIES),
        help=(
            "Values converted to m=floor(p*n*(n-1)/2), then GenRndGnm "
            "generates exactly m edges. Multiple values are assigned to "
            "successive graphs in a cycle (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=sorted(algorithm_registry),
        default=list(algorithm_registry),
        help="Algorithms to benchmark.",
    )
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--random-seed", type=snap_seed, default=42)
    parser.add_argument(
        "--ant-rlf-iterations",
        "--ant-iterations",
        dest="ant_iterations",
        type=positive_int,
        default=20,
    )
    parser.add_argument(
        "--ant-rlf-ants",
        "--ants-per-iteration",
        dest="ants_per_iteration",
        type=positive_int,
        default=10,
    )
    parser.add_argument("--tabu-iterations", type=positive_int, default=2000)
    parser.add_argument("--tabu-restarts", type=positive_int, default=5)
    parser.add_argument("--max-colors", type=positive_int, default=None)
    parser.add_argument("--min-colors", type=positive_int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-coloring", action="store_true")
    parser.add_argument("--coloring-print-limit", type=positive_int, default=200)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        snap_module = load_snap()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    algorithm_registry = get_random_algorithm_registry()
    graph_specs = build_graph_specs(args)
    ensure_dir(args.graph_dir)
    ensure_dir(args.result_path.parent)

    report: Dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": "Synthetic random graphs generated by SNAP",
        "graph_generator": (
            "snap.GenRndGnm(snap.TUNGraph, n, m, False, snap.TRnd(seed))"
        ),
        "graph_model": "G(n,m)",
        "graph_dir": str(args.graph_dir),
        "graph_count": args.graph_count,
        "random_node_count_range": [MIN_RANDOM_NODES, MAX_RANDOM_NODES],
        "generated_node_counts": [spec.node_count for spec in graph_specs],
        "edge_probabilities": list(args.edge_probabilities),
        "base_random_seed": args.random_seed,
        "algorithms": list(args.algorithms),
        "coloring_game_parameter_policy": {
            "policy": "class defaults only",
            "node_mode": 0,
            "traverse_mode": "forward",
            "color_choice_mode": "forward",
            "random_seed": 42,
        },
        "execution_time_scope": (
            "Algorithm construction, graph loading, solving, and coloring "
            "validation; one-off module imports are excluded."
        ),
        "datasets": [],
    }

    for spec in graph_specs:
        graph_path, stats = generate_random_graph(
            snap_module,
            spec,
            args.graph_dir,
        )
        graph_report: Dict[str, object] = {
            "dataset": spec.graph_id,
            "description": "SNAP random undirected G(n,m) graph.",
            "graph_path": str(graph_path),
            **stats,
            "results": [],
        }

        print(
            f"Graph {spec.graph_id}: nodes={stats['nodes']}, "
            f"edges={stats['edges']}, graph={graph_path}"
        )
        run_args = argparse.Namespace(**vars(args))
        run_args.random_seed = spec.seed
        context = RunContext(graph_path=graph_path.resolve(), args=run_args)

        for algorithm_name in args.algorithms:
            algorithm = algorithm_registry[algorithm_name]
            with temporary_constructor_graph(algorithm_name):
                result = run_algorithm(algorithm, context)
            graph_report["results"].append(result)
            print_algorithm_result(result, args)

        report["datasets"].append(graph_report)

    with args.result_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Full random-graph benchmark results saved to {args.result_path}")
    print(f"Generated graph structures saved under {args.graph_dir}")

    if not args.no_plots:
        generated_plots = plot_benchmark_images(report, args.plot_dir)
        for plot_path in generated_plots:
            print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
