"""
Evaluator for the (n,w)-admissible set example.

This evaluator imports the evolved module (the program) and calls
`generate_A(n, w)` to obtain a candidate set. It verifies the three
conditions (constant weight, unique support, triple-existence), and returns
metrics including a `combined_score` used by OpenEvolve.

Metric design (simple):
- `valid_fraction`: fraction of vectors that pass constant-weight & value checks
- `unique_support_ok`: 1.0 if supports are unique, else 0.0
- `triple_ok_fraction`: fraction of tested triples that satisfy the 3-item condition
- `size`: cardinality of A (encouraged to be large)
- `combined_score`: weighted aggregate in [0,1]
"""

from typing import List, Tuple, Dict, Any, Optional
import importlib.util
import sys
import os
import itertools
import json


class TunableManager:
    """Manage tunable calls during module execution.

    Modes:
    - 'capture': record each tunable(options) call and return a proxy object.
    - 'assign': return concrete choice according to provided assignment list; if
      a call index is not assigned, return the first option as default.
    """

    def __init__(self, mode: str = "capture", assignment: Optional[List[int]] = None):
        self.mode = mode
        self.assignment = assignment or []
        self.calls: List[List[Any]] = []
        self._call_index = 0

    def tunable(self, options):
        # ensure options is a sequence
        opts = list(options)
        if self.mode == "capture":
            self.calls.append(opts)
            idx = len(self.calls) - 1
            # return a simple proxy object so callers who embed it into structures
            # get something printable; evaluator will re-run module to obtain concrete values
            return {"__tunable__": True, "id": idx, "options": opts}
        else:
            # assign mode: return assigned choice if available, else default to opts[0]
            ci = self._call_index
            self._call_index += 1
            if ci < len(self.assignment):
                sel = int(self.assignment[ci])
                if 0 <= sel < len(opts):
                    return opts[sel]
            return opts[0]

    def reset(self):
        self._call_index = 0


N = 12
W = 7


def _is_valid_vector(vec: List[int], n: int, w: int) -> bool:
    if len(vec) != n:
        return False
    for x in vec:
        if x not in (0, 1, 2):
            return False
    return sum(1 for x in vec if x != 0) == w

def _support(vec: List[int]) -> Tuple[int, ...]:
    return tuple(i for i, x in enumerate(vec) if x != 0)

def _triple_satisfies(a: List[int], b: List[int], c: List[int]) -> bool:
    # First try: load precomputed incompatible unordered pattern triples once
    try:
        _INCOMPAT_UNORDERED
    except NameError:
        _INCOMPAT_UNORDERED = set()
        try:
            compat_path = os.path.join(os.path.dirname(__file__), 'pattern_triple_compat.json')
            if os.path.exists(compat_path):
                with open(compat_path, 'r', encoding='utf-8') as f:
                    j = json.load(f)
                    inc = j.get('incompatible', [])
                    for t in inc:
                        _INCOMPAT_UNORDERED.add(tuple(sorted(t)))
        except Exception:
            _INCOMPAT_UNORDERED = set()

        # If we failed to load a precompute file, try to load the user's
        # `test_symmetry_admissible_set_opt.py` bad_triples as a fallback.
        if not _INCOMPAT_UNORDERED:
            try:
                ext_path = os.path.join('D:/Lab_ML/x-evolve-main', 'test_symmetry_admissible_set_opt.py')
                if os.path.exists(ext_path):
                    import ast as _ast
                    with open(ext_path, 'r', encoding='utf-8') as ef:
                        src = ef.read()
                    start = src.find('bad_triples = set(')
                    if start != -1:
                        open_paren = src.find('set(', start)
                        open_bracket = src.find('[', open_paren)
                        close_bracket = src.find('])', open_bracket)
                        if open_bracket != -1 and close_bracket != -1:
                            list_text = src[open_bracket:close_bracket+1]
                            pylist = _ast.literal_eval(list_text)
                            for t in pylist:
                                _INCOMPAT_UNORDERED.add(tuple(sorted(t)))
            except Exception:
                pass

    # Group-level quick check: group size = 3
    n = len(a)
    group_size = 3
    num_groups = (n + group_size - 1) // group_size

    # helper to map a group's values to a pattern id (0..6) if possible
    PATTERNS = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 2),
        (0, 2, 1),
        (1, 1, 1),
        (2, 2, 2),
    ]

    def group_pattern_id(vec, gi):
        start = gi * group_size
        end = min(start + group_size, n)
        seg = tuple(vec[start:end])
        # try match seg to any rotation of patterns (truncated for last group)
        for pid, pat in enumerate(PATTERNS):
            # generate rotations
            rots = [(pat[0], pat[1], pat[2]), (pat[2], pat[0], pat[1]), (pat[1], pat[2], pat[0])]
            for r in rots:
                if seg == tuple(r[: end - start]):
                    return pid
        return None

    # If we have the unordered incompatible set, and every group maps to a pattern
    # and the sorted pid-triple for every group is in the set, then there is no
    # coordinate that can satisfy the allowed multisets -> triple fails.
    if _INCOMPAT_UNORDERED:
        all_groups_incompat = True
        for gi in range(num_groups):
            p1 = group_pattern_id(a, gi)
            p2 = group_pattern_id(b, gi)
            p3 = group_pattern_id(c, gi)
            if p1 is None or p2 is None or p3 is None:
                all_groups_incompat = False
                break
            if tuple(sorted((p1, p2, p3))) not in _INCOMPAT_UNORDERED:
                all_groups_incompat = False
                break
        if all_groups_incompat:
            return False

    # Fallback: exact coordinate-wise check
    for i in range(len(a)):
        vals = sorted([a[i], b[i], c[i]])
        if vals == [0, 0, 1] or vals == [0, 0, 2] or vals == [0, 1, 2]:
            return True
    return False


def evaluate(program_path: str) -> Dict[str, Any]:
    """Load the program file and evaluate the returned candidate set.

    Returns a dict of metrics. OpenEvolve expects some numeric metrics such as
    `combined_score` to drive evolution.
    """
    if not os.path.exists(program_path):
        return {"error": 0.0}

    # helper: run module with a TunableManager and return (A, manager, exec_error)
    def run_module_with_manager(assignment: Optional[List[int]] = None, capture: bool = False):
        manager = TunableManager(mode=("capture" if capture else "assign"), assignment=(assignment or []))
        spec = importlib.util.spec_from_file_location("candidate_module", program_path)
        if spec is None or spec.loader is None:
            return None, manager, "spec_error"
        module = importlib.util.module_from_spec(spec)
        # inject the tunable function into module globals before executing
        module.tunable = manager.tunable
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            return None, manager, str(e)
        if not hasattr(module, "generate_A"):
            return None, manager, "module missing generate_A"
        try:
            manager.reset()
            A = module.generate_A(N, W)
        except Exception as e:
            return None, manager, str(e)
        return A, manager, None

    # helper: evaluate a returned A (existing logic moved here)
    def _score_from_A(A) -> Dict[str, Any]:
        # returns metrics dict similar to previous return
        # Handle grouped and non-grouped formats
        vectors: List[List[int]] = []
        if isinstance(A, dict) and A.get("grouped"):
            n = int(A.get("n", N))
            w = int(A.get("w", W))
            group_patterns = list(A.get("group_patterns", []))

            PATTERNS = [
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 2),
                (0, 1, 2),
                (0, 2, 1),
                (1, 1, 1),
                (2, 2, 2),
            ]

            group_size = 3
            num_groups = len(group_patterns)

            per_group_assignments = []
            for gi, pid in enumerate(group_patterns):
                pid = int(pid)
                pattern = PATTERNS[pid]
                rotations = []
                a, b, c = pattern
                rotations.append((a, b, c))
                rotations.append((c, a, b))
                rotations.append((b, c, a))
                seen = set()
                uniq = []
                for r in rotations:
                    if r not in seen:
                        seen.add(r)
                        uniq.append(r)
                start = gi * group_size
                end = min(start + group_size, n)
                g_len = end - start
                if g_len < 3:
                    assignments = [tuple(r[:g_len]) for r in uniq]
                else:
                    assignments = list(uniq)
                per_group_assignments.append(assignments)

            def nonzeros_in_assignment(assign):
                return sum(1 for x in assign if x != 0)

            per_group_nz = [[nonzeros_in_assignment(a) for a in assigns] for assigns in per_group_assignments]

            dp = {0: 1}
            for assigns_nz in per_group_nz:
                ndp = {}
                for cur_sum, cnt in dp.items():
                    for nz in assigns_nz:
                        ndp[cur_sum + nz] = ndp.get(cur_sum + nz, 0) + cnt
                dp = ndp
            total_after_prune = dp.get(w, 0)

            EXPAND_THRESHOLD = 5000
            SAMPLE_VECTOR_LIMIT = 2000

            if total_after_prune == 0:
                return {"size": 0, "valid_count": 0, "unique_support_ok": 0.0, "triple_ok_fraction": 0.0, "combined_score": 0.0}

            if total_after_prune <= EXPAND_THRESHOLD:
                for combo in itertools.product(*[range(len(a)) for a in per_group_assignments]):
                    nzsum = 0
                    for gi, ai in enumerate(combo):
                        nzsum += per_group_nz[gi][ai]
                        if nzsum > w:
                            break
                    if nzsum != w:
                        continue
                    vec = [0] * n
                    for gi, ai in enumerate(combo):
                        start = gi * group_size
                        assigns = per_group_assignments[gi][ai]
                        for offset, val in enumerate(assigns):
                            vec[start + offset] = val
                    vectors.append(vec)
            else:
                import random

                rng = random.Random(0)
                samples = set()

                def dfs_sample(gi, cur_vec, cur_nz):
                    if len(samples) >= SAMPLE_VECTOR_LIMIT:
                        return
                    if gi == num_groups:
                        if cur_nz == w:
                            samples.add(tuple(cur_vec))
                        return
                    assigns = per_group_assignments[gi]
                    idxs = list(range(len(assigns)))
                    rng.shuffle(idxs)
                    for ai in idxs:
                        nz = per_group_nz[gi][ai]
                        if cur_nz + nz > w:
                            continue
                        start = gi * group_size
                        new_vec = list(cur_vec)
                        for offset, val in enumerate(assigns[ai]):
                            new_vec[start + offset] = val
                        dfs_sample(gi + 1, new_vec, cur_nz + nz)

                dfs_sample(0, [0] * n, 0)
                vectors = [list(v) for v in list(samples)[:SAMPLE_VECTOR_LIMIT]]

        else:
            if not isinstance(A, list):
                return {"error": 0.0, "error_msg": "generate_A returned unsupported type"}
            vectors = A

        total = len(vectors)
        valid_count = 0
        supports = {}
        for vec in vectors:
            if _is_valid_vector(vec, N, W):
                valid_count += 1
            s = _support(vec)
            supports.setdefault(s, 0)
            supports[s] += 1

        unique_support_ok = 1.0 if all(c == 1 for c in supports.values()) else 0.0

        triples = list(itertools.combinations(vectors, 3))
        if len(triples) == 0:
            triple_ok_fraction = 1.0
        else:
            max_check = 200
            if len(triples) > max_check:
                k = max(1, len(triples) // max_check)
                check_triples = [triples[i] for i in range(0, len(triples), k)][:max_check]
            else:
                check_triples = triples

            ok = 0
            for a, b, c in check_triples:
                if _triple_satisfies(a, b, c):
                    ok += 1
            triple_ok_fraction = ok / len(check_triples)

        if total == 0:
            combined = 0.0
        else:
            size_score = min(total / (1.0 * (len(list(itertools.combinations(range(N), W))) or 1)), 1.0)
            combined = 0.5 * size_score + 0.3 * (valid_count / max(1, total)) + 0.2 * triple_ok_fraction

        return {
            "size": total,
            "valid_count": valid_count,
            "unique_support_ok": unique_support_ok,
            "triple_ok_fraction": triple_ok_fraction,
            "combined_score": float(combined),
        }

    # First run in capture mode to detect tunables
    A_cap, manager_cap, err = run_module_with_manager(assignment=None, capture=True)
    if err is not None:
        return {"error": 0.0, "exec_error": err}

    num_tunables = len(manager_cap.calls)
    # No tunables: evaluate directly
    if num_tunables == 0:
        return _score_from_A(A_cap)

    # Beam search over tunables to find good assignment. Beam width limits search.
    BEAM_WIDTH = 8
    MAX_EVALS = 200

    # helper to evaluate an assignment (list of choice indices)
    eval_cache = {}

    def eval_assignment(assign_tuple):
        if assign_tuple in eval_cache:
            return eval_cache[assign_tuple]
        A_run, mgr, err = run_module_with_manager(assignment=list(assign_tuple), capture=False)
        if err is not None or A_run is None:
            score = -1e9
            metrics = {"combined_score": -1e9}
        else:
            metrics = _score_from_A(A_run)
            score = float(metrics.get("combined_score", -1e9))
        eval_cache[assign_tuple] = (score, metrics)
        return eval_cache[assign_tuple]

    # initial beam: empty assignment
    beam = [tuple()]
    best_score = -1e9
    best_metrics = None
    evals = 0

    for depth in range(num_tunables):
        candidates = []
        for partial in beam:
            # determine number of options for this tunable from captured calls
            opts = manager_cap.calls[depth]
            for choice_idx in range(len(opts)):
                new_assign = tuple(list(partial) + [choice_idx])
                score, metrics = eval_assignment(new_assign)
                evals += 1
                candidates.append((score, new_assign, metrics))
                if score > best_score and len(new_assign) == num_tunables:
                    best_score = score
                    best_metrics = metrics
                if evals >= MAX_EVALS:
                    break
            if evals >= MAX_EVALS:
                break
        # keep top BEAM_WIDTH candidates by score
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = [c[1] for c in candidates[:BEAM_WIDTH]]
        # if any beam entry is full assignment, update best
        for score, assign, metrics in candidates:
            if len(assign) == num_tunables and score > best_score:
                best_score = score
                best_metrics = metrics
        if evals >= MAX_EVALS:
            break

    # If best_metrics not set (e.g., beam didn't reach full assignment), evaluate remaining beam fully
    if best_metrics is None:
        for assign in beam:
            if len(assign) < num_tunables:
                # expand greedily with zeros
                assign_full = tuple(list(assign) + [0] * (num_tunables - len(assign)))
            else:
                assign_full = assign
            score, metrics = eval_assignment(assign_full)
            if score > best_score:
                best_score = score
                best_metrics = metrics

    return best_metrics or {"error": 0.0}


if __name__ == "__main__":
    # Local quick check: import example initial program
    import json
    from initial_program import generate_A

    A = generate_A(N, W)
    print(json.dumps(evaluate(__file__), indent=2))
