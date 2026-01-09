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
        }
        if params:
            self.params.update(params)

    def _compute_priority(
        self,
        vec: Tuple[int, ...],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> float:
        """
        Compute intrinsic priority for the vector.
        """
        bw = params.get("balance_weight", 0.6)

        c0 = vec.count(0)
        c1 = vec.count(1)
        c2 = vec.count(2)
        imbalance = abs(c0 - c1) + abs(c1 - c2) + abs(c0 - c2)

        score = -bw * imbalance
        return score

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

        candidates: List[Tuple[int, ...]] = [v for v in all_vectors]
        priorities: Dict[Tuple[int, ...], float] = {
            v: self._compute_priority(v, active_params, rng) for v in candidates
        }
        sorted_candidates = sorted(candidates, key=lambda v: priorities.get(v, float("-inf")), reverse=True)

        for vec in sorted_candidates:
            if can_add(vec, capset, capset_set):
                capset.append(vec)
                capset_set.add(vec)

        return capset

    def tune_with_optuna(
        self,
        max_trials: int = 300,
        timeout: float = 180.0,
    ) -> Tuple[Dict[str, Any], List[Tuple[int, ...]]]:
        import optuna
        
        def objective(trial: Any) -> float:
            sampled = {
                "balance_weight": trial.suggest_float("balance_weight", 0.0, 2.0),
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
    best_params, tuned_capset = generator.tune_with_optuna(max_trials=300, timeout=180.0)
    return tuned_capset