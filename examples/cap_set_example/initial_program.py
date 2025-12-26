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
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Dict[Tuple[int, ...], float]:
        """
        Compute intrinsic priorities for candidates.

        This priority function intentionally does NOT depend on the current
        `capset`. It scores each candidate using only features intrinsic to
        the vector (e.g. coordinate imbalance) and optional random noise.

        The generator will compute these priorities once at the start,
        sort candidates by priority, then greedily attempt to add each
        candidate in order if it does not violate the cap set constraint.
        """
        priorities: Dict[Tuple[int, ...], float] = {}

        noise = params.get("random_noise", 0.0)
        bw = params.get("balance_weight", 0.6)

        for vec in candidates:
            # Intrinsic imbalance score (lower imbalance preferred)
            c0 = vec.count(0)
            c1 = vec.count(1)
            c2 = vec.count(2)
            imbalance = abs(c0 - c1) + abs(c1 - c2) + abs(c0 - c2)

            score = -bw * imbalance
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

        # Compute priorities once at the start without depending on the
        # currently selected cap set. Then greedily iterate the sorted list
        # and add any vector that preserves the cap-set property.
        candidates: List[Tuple[int, ...]] = [v for v in all_vectors]
        priorities = self._compute_priorities(candidates, active_params, rng)
        sorted_candidates = sorted(candidates, key=lambda v: priorities.get(v, float("-inf")), reverse=True)

        # Greedily accept vectors in sorted order if they keep the cap property
        for vec in sorted_candidates:
            if can_add(vec, capset, capset_set):
                capset.append(vec)
                capset_set.add(vec)

        return capset

    def tune_with_optuna(
        self,
        max_trials: int = 200,
        timeout: float = 180.0,
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
    """Entry point called by the evaluator: run tuning then return best cap set."""
    generator = GreedyCapSetGenerator(n=n)
    best_params, tuned_capset = generator.tune_with_optuna(max_trials=200, timeout=180.0)
    return tuned_capset

