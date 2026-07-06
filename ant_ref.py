"""
Ant-Ref graph coloring.

Idea:
1. Build one color class at a time (RLF-style maximal independent set).
2. While selecting the next vertex in the current color class, combine:
   - RLF heuristic signal
   - pheromone signal (how suitable a vertex is with current class)
3. Run multiple ants and iterations, evaporate + reinforce pheromone,
   keep the best coloring found.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

from utils import TEMPLATE_GRAPH_PATH, Undirected_graph


class Ant_Ref(Undirected_graph):
    def __init__(
        self,
        graph_name: str = "ant_ref",
        node_num: int = 5,
        edge_prob: float = 0.5,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.15,
        q: float = 1.0,
        ants_per_iteration: int = 20,
        max_iterations: int = 60,
        random_seed: int = 42,
    ):
        super().__init__(graph_name, node_num, edge_prob)

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.ants_per_iteration = ants_per_iteration
        self.max_iterations = max_iterations
        self.rng = random.Random(random_seed)

        self.node_ids: List[int] = []
        self.adj: Dict[int, Set[int]] = {}
        self.pheromone: Dict[Tuple[int, int], float] = {}

        self.best_coloring: Dict[int, int] = {}
        self.best_color_classes: List[List[int]] = []
        self.best_color_count: int = 10**9

        self._rebuild_internal_state()

    def _rebuild_internal_state(self) -> None:
        self.node_ids = sorted(node.GetId() for node in self.graph.Nodes())
        self.node_num = len(self.node_ids)

        self.adj = {nid: set() for nid in self.node_ids}
        for nid in self.node_ids:
            ni = self.graph.GetNI(nid)
            for i in range(ni.GetDeg()):
                nbr = ni.GetOutNId(i)
                self.adj[nid].add(nbr)

        self.node_color = {nid: 0 for nid in self.node_ids}
        self._init_pheromone()

    def _init_pheromone(self, initial: float = 1.0) -> None:
        self.pheromone = {}
        for i, u in enumerate(self.node_ids):
            for v in self.node_ids[i + 1 :]:
                if v not in self.adj[u]:
                    self.pheromone[(u, v)] = initial

    @staticmethod
    def _pair_key(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def _degree_in_subset(self, node_id: int, subset: Set[int]) -> int:
        return sum(1 for v in self.adj[node_id] if v in subset)

    def _get_pheromone(self, u: int, v: int) -> float:
        key = self._pair_key(u, v)
        return self.pheromone.get(key, 0.0)

    def _roulette_select(self, weighted_nodes: List[Tuple[int, float]]) -> int:
        positive = [(nid, max(weight, 0.0)) for nid, weight in weighted_nodes]
        total = sum(weight for _, weight in positive)
        if total <= 0:
            return self.rng.choice([nid for nid, _ in positive])

        r = self.rng.random() * total
        acc = 0.0
        for nid, weight in positive:
            acc += weight
            if acc >= r:
                return nid
        return positive[-1][0]

    def _select_seed(self, uncolored: Set[int]) -> int:
        weighted = []
        for v in uncolored:
            deg = self._degree_in_subset(v, uncolored)
            # RLF-inspired: high residual degree is a strong seed.
            score = (deg + 1.0) ** self.beta
            weighted.append((v, score))
        return self._roulette_select(weighted)

    def _select_candidate(
        self,
        candidates: List[int],
        color_class: List[int],
        uncolored: Set[int],
    ) -> int:
        candidate_set = set(candidates)
        weighted: List[Tuple[int, float]] = []

        for v in candidates:
            # RLF-style heuristic: prefer vertices with more links to
            # forbidden area and fewer links inside candidate area.
            forbidden_neighbors = self._degree_in_subset(v, uncolored - candidate_set)
            candidate_neighbors = self._degree_in_subset(v, candidate_set)
            heuristic = forbidden_neighbors + 1.0 / (1.0 + candidate_neighbors)

            if color_class:
                pheromone_values = [
                    self._get_pheromone(v, u) for u in color_class if self._get_pheromone(v, u) > 0
                ]
                pheromone = sum(pheromone_values) / len(pheromone_values) if pheromone_values else 1e-6
            else:
                pheromone = 1.0

            score = (pheromone**self.alpha) * (heuristic**self.beta)
            weighted.append((v, score))

        return self._roulette_select(weighted)

    def _construct_color_class(self, uncolored: Set[int]) -> List[int]:
        color_class: List[int] = []

        while uncolored:
            if not color_class:
                seed = self._select_seed(uncolored)
                color_class.append(seed)
                uncolored.remove(seed)
                continue

            candidates = [
                v
                for v in uncolored
                if all((u not in self.adj[v]) for u in color_class)
            ]
            if not candidates:
                break

            selected = self._select_candidate(candidates, color_class, uncolored)
            color_class.append(selected)
            uncolored.remove(selected)

        return color_class

    def _construct_one_solution(self) -> Tuple[Dict[int, int], int, List[List[int]]]:
        uncolored = set(self.node_ids)
        color_id = 1
        coloring: Dict[int, int] = {}
        color_classes: List[List[int]] = []

        while uncolored:
            color_class = self._construct_color_class(uncolored)
            for node_id in color_class:
                coloring[node_id] = color_id
            color_classes.append(color_class)
            color_id += 1

        return coloring, len(color_classes), color_classes

    def _build_color_classes(self, coloring: Dict[int, int]) -> List[List[int]]:
        bucket: Dict[int, List[int]] = {}
        for node_id, color_id in coloring.items():
            bucket.setdefault(color_id, []).append(node_id)
        return [bucket[c] for c in sorted(bucket)]

    def _is_valid_coloring(self, coloring: Dict[int, int]) -> bool:
        for u in self.node_ids:
            for v in self.adj[u]:
                if u < v and coloring.get(u) == coloring.get(v):
                    return False
        return True

    def _update_pheromone(
        self,
        iter_best_color_classes: List[List[int]],
        iter_best_color_count: int,
    ) -> None:
        # Evaporation
        for key in list(self.pheromone.keys()):
            self.pheromone[key] = max(1e-6, (1.0 - self.rho) * self.pheromone[key])

        # Reinforcement for the best solution in this iteration
        delta = self.q / max(iter_best_color_count, 1)
        for color_class in iter_best_color_classes:
            for i, u in enumerate(color_class):
                for v in color_class[i + 1 :]:
                    key = self._pair_key(u, v)
                    if key in self.pheromone:
                        self.pheromone[key] += delta

        # Extra reinforcement for global best (stability)
        if self.best_coloring:
            best_delta = self.q / max(self.best_color_count, 1)
            for color_class in self.best_color_classes:
                for i, u in enumerate(color_class):
                    for v in color_class[i + 1 :]:
                        key = self._pair_key(u, v)
                        if key in self.pheromone:
                            self.pheromone[key] += best_delta

    def solve(
        self,
        max_iterations: int | None = None,
        ants_per_iteration: int | None = None,
        verbose: bool = False,
    ) -> Tuple[Dict[int, int], int]:
        max_iterations = max_iterations or self.max_iterations
        ants_per_iteration = ants_per_iteration or self.ants_per_iteration

        self.best_coloring = {}
        self.best_color_classes = []
        self.best_color_count = 10**9

        for iteration in range(1, max_iterations + 1):
            iter_best_coloring: Dict[int, int] = {}
            iter_best_color_count = 10**9
            iter_best_color_classes: List[List[int]] = []

            for _ in range(ants_per_iteration):
                coloring, color_count, color_classes = self._construct_one_solution()
                if not self._is_valid_coloring(coloring):
                    continue

                if color_count < iter_best_color_count:
                    iter_best_coloring = coloring
                    iter_best_color_count = color_count
                    iter_best_color_classes = color_classes

            if not iter_best_coloring:
                raise RuntimeError("Failed to construct a feasible coloring.")

            if iter_best_color_count < self.best_color_count:
                self.best_coloring = dict(iter_best_coloring)
                self.best_color_count = iter_best_color_count
                self.best_color_classes = [list(cls) for cls in iter_best_color_classes]

            self._update_pheromone(iter_best_color_classes, iter_best_color_count)

            if verbose:
                print(
                    f"[Iter {iteration}] iter_best={iter_best_color_count}, "
                    f"global_best={self.best_color_count}"
                )

        self.node_color = dict(self.best_coloring)
        return dict(self.best_coloring), self.best_color_count

    def color_graph(
        self,
        max_iterations: int | None = None,
        ants_per_iteration: int | None = None,
        verbose: bool = False,
    ) -> Tuple[Dict[int, int], int]:
        return self.solve(max_iterations=max_iterations, ants_per_iteration=ants_per_iteration, verbose=verbose)

    def load_graph(self, file_path: str) -> None:
        super().load_graph(file_path)
        self._rebuild_internal_state()

    def print_result(self) -> None:
        if not self.node_color:
            print("No coloring result.")
            return

        used_colors = sorted(set(self.node_color.values()))
        print(f"colors used: {len(used_colors)}")
        for node_id in sorted(self.node_color):
            print(f"node {node_id}, color {self.node_color[node_id]}")


def main() -> None:
    agent = Ant_Ref(
        ants_per_iteration=30,
        max_iterations=80,
        random_seed=7,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q=1.0,
    )
    agent.load_graph(TEMPLATE_GRAPH_PATH)
    agent.solve(verbose=True)
    agent.print_result()


if __name__ == "__main__":
    main()
