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
            # Fewer knobs for a lightweight demo; keep the interface intact.
            "balance_weight": 0.6,
            "transition_penalty": 0.4,
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

    def _priority(self, vec: Tuple[int, ...], params: Dict[str, Any], rng: random.Random) -> float:
        """
        Score a vector using only two simple signals so the demo stays easy to follow:
        - how balanced the digit counts are (favoring even mixtures)
        - how often identical digits appear back-to-back (penalizing long runs)
        Small noise keeps ordering deterministic with the seeded RNG.
        """
        count0 = vec.count(0)
        count1 = vec.count(1)
        count2 = vec.count(2)
        imbalance = abs(count0 - count1) + abs(count1 - count2) + abs(count0 - count2)
        transitions = sum(1 for a, b in zip(vec, vec[1:]) if a == b)

        score = -params["balance_weight"] * imbalance - params["transition_penalty"] * transitions
        noise = params.get("random_noise", 0)
        return score + (noise * rng.random() if noise else 0.0)

    def generate(self, params: Optional[Dict[str, Any]] = None, rng: Optional[random.Random] = None) -> List[Tuple[int, ...]]:
        """
        Greedy cap-set construction with a minimal priority function. The vector pool
        is shuffled via the priority score to keep the demo deterministic yet simple.
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
        Simplified stub for demonstration: keep the interface but skip heavy tuning.
        If Optuna is installed, we still avoid running it to keep runtime predictable.
        """
        return dict(self.params), self.generate(params=self.params, rng=random.Random(_RANDOM_SEED))


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

