from typing import List, Tuple, Dict, Any, Optional
import itertools
import random

N = 8
W = 4

class GreedyAdmissibleGenerator:
    """
    Greedy generator for (n,w)-admissible sets with tunable priority weights.
    """

    def __init__(self, n: int = N, w: int = W, params: Optional[Dict[str, Any]] = None):
        self.n = n
        self.w = w
        self.params: Dict[str, Any] = {
            "two_weight": 3.0,
            "one_weight": 1.0,
        }
        if params:
            self.params.update(params)

    def _compute_priority(self, vec: Tuple[int, ...], params: Dict[str, Any], rng: random.Random) -> float:
        twos = vec.count(2)
        ones = vec.count(1)
        score = params.get("two_weight", 3.0) * twos + params.get("one_weight", 1.0) * ones
        return score

    @staticmethod
    def _is_valid_vector(vec: Tuple[int, ...], n: int, w: int) -> bool:
        if len(vec) != n:
            return False
        for x in vec:
            if x not in (0, 1, 2):
                return False
        return sum(1 for x in vec if x != 0) == w

    @staticmethod
    def _triple_satisfies(a: Tuple[int, ...], b: Tuple[int, ...], c: Tuple[int, ...]) -> bool:
        if not (len(a) == len(b) == len(c)):
            return False
        for i in range(len(a)):
            vals = sorted((a[i], b[i], c[i]))
            if vals == [0, 0, 1] or vals == [0, 0, 2] or vals == [0, 1, 2]:
                return True
        return False

    def generate(self, params: Optional[Dict[str, Any]] = None, rng: Optional[random.Random] = None) -> List[Tuple[int, ...]]:
        if rng is None:
            rng = random.Random(0)
        active_params = dict(self.params)
        if params:
            active_params.update(params)

        all_vectors = (v for v in itertools.product((0, 1, 2), repeat=self.n) if sum(1 for x in v if x != 0) == self.w)
        candidates = list(all_vectors)

        priorities = {v: self._compute_priority(v, active_params, rng) for v in candidates}
        sorted_candidates = sorted(candidates, key=lambda v: (priorities.get(v, 0.0), v), reverse=True)

        A: List[Tuple[int, ...]] = []
        supports = set()

        for vec in sorted_candidates:
            supp = tuple(i for i, val in enumerate(vec) if val != 0)
            if supp in supports:
                continue
            if not self._is_valid_vector(vec, self.n, self.w):
                continue
            ok = True
            if len(A) >= 2:
                for x, y in itertools.combinations(A, 2):
                    if not self._triple_satisfies(x, y, vec):
                        ok = False
                        break
            if ok:
                A.append(vec)
                supports.add(supp)

        return A

    def tune_with_optuna(self, max_trials: int = 300, timeout: float = 180.0) -> Tuple[Dict[str, Any], List[Tuple[int, ...]]]:
        try:
            import optuna
        except Exception as e:
            raise RuntimeError("optuna is required for tuning but not installed") from e

        try:
            import logging
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            logging.getLogger("optuna").setLevel(logging.WARNING)
        except Exception:
            pass

        def objective(trial: Any) -> float:
            sampled = {
                "two_weight": trial.suggest_float("two_weight", 0.0, 6.0),
                "one_weight": trial.suggest_float("one_weight", 0.0, 4.0),
            }
            rng = random.Random(0)
            A = self.generate(params=sampled, rng=rng)
            return float(len(A))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=max_trials, timeout=timeout)

        best_params = dict(self.params)
        best_params.update(study.best_params)
        best_set = self.generate(params=best_params, rng=random.Random(0))
        return best_params, best_set

def generate_A(n: int = N, w: int = W, seed: int = 0) -> List[List[int]]:
    gen = GreedyAdmissibleGenerator(n=n, w=w)
    best_params, best_set = gen.tune_with_optuna(max_trials=100, timeout=60.0)
    return [list(v) for v in best_set]