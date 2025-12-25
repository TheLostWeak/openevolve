import itertools
import random
from typing import Tuple, List, Optional, Dict, Any

# Hard-coded dimension for this example. All generator code should use this value.
CAP_N = 8

# Fixed random seed for reproducibility.
_RANDOM_SEED = 42


def can_add(new_vec: Tuple[int, ...], existing_list: List[Tuple[int, ...]], existing_set: Optional[set] = None) -> bool:
    """
    Efficient incremental check for whether `new_vec` can be added to the
    current cap set without violating the cap set property.

    Returns:
        True if `new_vec` can be added without creating a 3-term arithmetic
        progression (mod 3), otherwise False.
    """
    if existing_set is None:
        existing_set = set(existing_list)

    # Duplicate check
    if new_vec in existing_set:
        return False

    for x in existing_list:
        y = tuple((-(a + b)) % 3 for a, b in zip(x, new_vec))
        if y in existing_set and y != x:
            return False

    return True


class GreedyCapSetGenerator:
    """
    Class wrapper so LLMs can expose tunable parameters and search strategies.
    The priority function is parameterized; a tuning hook can adjust weights
    before building the final cap set.
    """

    def __init__(self, n: int = CAP_N, params: Optional[Dict[str, Any]] = None):
        self.n = n
        self.params: Dict[str, Any] = {
            "weight_zero": 1.0,
            "weight_one": 1.1,
            "weight_two": 0.9,
            "spread_weight": 0.25,
            "adjacent_penalty": 0.15,
            "random_noise": 0.05,
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

    def _priority(self, vec: Tuple[int, ...], params: Dict[str, Any], rng: random.Random) -> float:
        """
        Score a vector using lightweight, tunable features:
        - counts of each digit
        - spread of positions (favor balanced mixes)
        - small noise to break ties deterministically via seeded RNG
        """
        count0 = vec.count(0)
        count1 = vec.count(1)
        count2 = vec.count(2)
        diff01 = abs(count0 - count1)
        diff12 = abs(count1 - count2)
        diff02 = abs(count0 - count2)

        # Adjacent transition penalty to encourage dispersed digits
        transitions = sum(1 for a, b in zip(vec, vec[1:]) if a == b)

        score = (
            params["weight_zero"] * count0
            + params["weight_one"] * count1
            + params["weight_two"] * count2
            - params["spread_weight"] * (diff01 + diff12 + diff02)
            - params["adjacent_penalty"] * transitions
        )
        if params.get("random_noise", 0) > 0:
            score += params["random_noise"] * rng.random()
        return score

    def generate(self, params: Optional[Dict[str, Any]] = None, rng: Optional[random.Random] = None) -> List[Tuple[int, ...]]:
        """
        Greedy cap-set construction with a tunable priority function.
        """
        if rng is None:
            rng = random.Random(_RANDOM_SEED)
        active_params = dict(self.params)
        if params:
            active_params.update(params)

        all_vectors = list(itertools.product([0, 1, 2], repeat=self.n))
        sorted_vectors = sorted(all_vectors, key=lambda v: self._priority(v, active_params, rng), reverse=True)

        capset: List[Tuple[int, ...]] = []
        capset_set = set()

        for vec in sorted_vectors:
            if vec in capset_set:
                continue
            if can_add(vec, capset, capset_set):
                capset.append(vec)
                capset_set.add(vec)

        return capset

    def tune_with_optuna(self, max_trials: int = 20, timeout: float = 8.0) -> Tuple[Dict[str, Any], Optional[List[Tuple[int, ...]]]]:
        """
        Optional Optuna tuner. Safely no-ops if Optuna is unavailable so the
        module stays runnable under stdlib-only constraints.
        """
        optuna = self._try_import_optuna()
        if optuna is None:
            return dict(self.params), None

        sampler = optuna.samplers.TPESampler(seed=_RANDOM_SEED)

        def objective(trial: "optuna.trial.Trial") -> float:  # type: ignore[name-defined]
            trial_params = {
                "weight_zero": trial.suggest_float("weight_zero", 0.6, 1.4),
                "weight_one": trial.suggest_float("weight_one", 0.6, 1.6),
                "weight_two": trial.suggest_float("weight_two", 0.6, 1.4),
                "spread_weight": trial.suggest_float("spread_weight", 0.05, 0.6),
                "adjacent_penalty": trial.suggest_float("adjacent_penalty", 0.05, 0.6),
                "random_noise": trial.suggest_float("random_noise", 0.0, 0.15),
            }
            local_rng = random.Random(_RANDOM_SEED + trial.number)
            capset = self.generate(params=trial_params, rng=local_rng)
            return float(len(capset))

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=max_trials, timeout=timeout, show_progress_bar=False)

        best_params = dict(self.params)
        best_params.update(study.best_params)
        # Rebuild with best params for reproducibility
        best_capset = self.generate(params=best_params, rng=random.Random(_RANDOM_SEED + 999))
        self.params = best_params
        return best_params, best_capset


def generate_set(n: int) -> List[Tuple[int, ...]]:
    """
    Wrapper used by the evaluator. Keeps CAP_N fixed, but exposes a tuning
    hook that is safe when Optuna is missing.
    """
    generator = GreedyCapSetGenerator(n=CAP_N)
    best_params, tuned_capset = generator.tune_with_optuna(max_trials=15, timeout=6.0)
    if tuned_capset is None:
        # Optuna unavailable; fall back to deterministic greedy build.
        return generator.generate(params=best_params)
    return tuned_capset


