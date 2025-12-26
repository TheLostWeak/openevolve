import itertools
import random
from typing import Tuple, List, Optional, Dict, Any, Set

CAP_N = 8
_RANDOM_SEED = 42


def can_add(
    new_vec: Tuple[int, ...],
    existing_list: List[Tuple[int, ...]],
    existing_set: Optional[set] = None,
) -> bool:
    if existing_set is None:
        existing_set = set(existing_list)

    if new_vec in existing_set:
        return False

    for x in existing_list:
        y = tuple((-(a + b)) % 3 for a, b in zip(x, new_vec))
        if y in existing_set and y != x:
            return False

    return True


class GreedyCapSetGenerator:
    def __init__(self, n: int = CAP_N, params: Optional[Dict[str, Any]] = None):
        self.n = n
        self.params: Dict[str, Any] = {
            "balance_weight": 0.6,
            "existing_penalty": 0.1,
            "random_noise": 0.02,
        }
        if params:
            self.params.update(params)

    @staticmethod
    def _try_import_optuna():
        try:
            import optuna  # type: ignore
        except ImportError:
            return None
        return optuna

    def _compute_priorities(
        self,
        candidates: List[Tuple[int, ...]],
        capset: List[Tuple[int, ...]],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Dict[Tuple[int, ...], float]:
        """
        Compute priorities for all candidates once per round and return them.

        Simple, example-level scoring that combines:
        - imbalance (intrinsic to the vector)
        - existing_penalty * len(capset) (weak coupling to already-selected set)
        - small random noise for tie-breaking (optional)
        """
        priorities: Dict[Tuple[int, ...], float] = {}

        existing_term = params["existing_penalty"] * len(capset)
        noise = params.get("random_noise", 0.0)
        bw = params["balance_weight"]

        for vec in candidates:
            c0 = vec.count(0)
            c1 = vec.count(1)
            c2 = vec.count(2)
            imbalance = abs(c0 - c1) + abs(c1 - c2) + abs(c0 - c2)

            score = -bw * imbalance - existing_term
            if noise:
                score += noise * rng.random()

            priorities[vec] = score

        return priorities

    def generate(
        self,
        params: Optional[Dict[str, Any]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[int, ...]]:
        if rng is None:
            rng = random.Random(_RANDOM_SEED)

        active_params = dict(self.params)
        if params:
            active_params.update(params)

        all_vectors = list(itertools.product([0, 1, 2], repeat=self.n))

        capset: List[Tuple[int, ...]] = []
        capset_set: Set[Tuple[int, ...]] = set()

        # Dynamic greedy: after adding a point, recompute priorities for all
        # candidates. The priority computation function is called once per round.
        while True:
            # First collect this round's candidates that can be added
            candidates: List[Tuple[int, ...]] = []
            for vec in all_vectors:
                if vec in capset_set:
                    continue
                if can_add(vec, capset, capset_set):
                    candidates.append(vec)

            if not candidates:
                break

            # Compute priorities in batch (called once per round)
            priorities = self._compute_priorities(candidates, capset, active_params, rng)

            # Then scan to pick the highest-scoring candidate
            best_vec = None
            best_score = float("-inf")
            for vec in candidates:
                s = priorities[vec]
                if s > best_score:
                    best_score = s
                    best_vec = vec

            if best_vec is None:
                break

            capset.append(best_vec)
            capset_set.add(best_vec)

        return capset

    def tune_with_optuna(
        self,
        max_trials: int = 20,
        timeout: float = 30.0,
    ) -> Tuple[Dict[str, Any], List[Tuple[int, ...]]]:
        optuna = self._try_import_optuna()
        if optuna is None:
            raise RuntimeError("Optuna is not installed. Please run: pip install optuna")

        def objective(trial: Any) -> float:
            sampled = {
                "balance_weight": trial.suggest_float("balance_weight", 0.0, 2.0),
                "existing_penalty": trial.suggest_float("existing_penalty", 0.0, 1.0),
                "random_noise": trial.suggest_float("random_noise", 0.0, 0.05),
            }
            cap = self.generate(params=sampled, rng=random.Random(_RANDOM_SEED))
            return float(len(cap))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=_RANDOM_SEED),
        )
        study.optimize(objective, n_trials=max_trials, timeout=timeout)

        best_params = dict(self.params)
        best_params.update(study.best_params)
        best_capset = self.generate(params=best_params, rng=random.Random(_RANDOM_SEED))
        return best_params, best_capset


def generate_set(n: int) -> List[Tuple[int, ...]]:
    generator = GreedyCapSetGenerator(n=CAP_N)
    best_params, tuned_capset = generator.tune_with_optuna(max_trials=100, timeout=30.0)
    return tuned_capset

