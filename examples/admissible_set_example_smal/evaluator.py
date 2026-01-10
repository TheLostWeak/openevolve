from typing import List, Tuple, Dict, Any, Optional
import importlib.util
import sys
import os
import itertools
import json

N = 8
W = 4

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
    if not (len(a) == len(b) == len(c)):
        return False
    for i in range(len(a)):
        vals = sorted((a[i], b[i], c[i]))
        if vals == [0, 0, 1] or vals == [0, 0, 2] or vals == [0, 1, 2]:
            return True
    return False

def evaluate(program_path: str) -> Dict[str, Any]:
    if not os.path.exists(program_path):
        return {"error": 0.0}

    def run_module(capture: bool = False):
        spec = importlib.util.spec_from_file_location("candidate_module", program_path)
        if spec is None or spec.loader is None:
            return None, "spec_error"
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            return None, str(e)
        if not hasattr(module, "generate_A"):
            return None, "module missing generate_A"
        try:
            A = module.generate_A(N, W)
        except Exception as e:
            return None, str(e)
        return A, None

    def _score_from_A(A) -> Dict[str, Any]:
        # returns metrics dict similar to previous return
        # Only accept list-of-vectors format. Reject any grouped/pattern dicts.
        if not isinstance(A, list):
            return {"error": 0.0, "error_msg": "generate_A must return a list of vectors (grouped/pattern formats not supported)"}
        vectors: List[List[int]] = A

        total = len(vectors)

        # Validate each vector strictly (length n, values in {0,1,2}, weight w)
        valid_flags = [_is_valid_vector(vec, N, W) for vec in vectors]
        valid_count = sum(1 for v in valid_flags if v)
        valid_all = (valid_count == total and total > 0) or (total == 0)
        valid_fraction = (valid_count / total) if total > 0 else 0.0

        # Unique support: only meaningful when all vectors are valid
        supports = {}
        for vec in vectors:
            s = _support(vec)
            supports.setdefault(s, 0)
            supports[s] += 1

        unique_support_ok = 1.0 if (valid_all and all(c == 1 for c in supports.values())) else 0.0

        # Triple condition: strict universal check over all distinct triples in A.
        # Only consider strict pass if all vectors valid and every triple satisfies.
        triple_ok_all = 1.0
        triple_ok_fraction = 1.0
        if not valid_all:
            triple_ok_all = 0.0
            triple_ok_fraction = 0.0
        else:
            triples = list(itertools.combinations(vectors, 3))
            if len(triples) == 0:
                triple_ok_all = 1.0
                triple_ok_fraction = 1.0
            else:
                ok = 0
                for a, b, c in triples:
                    if _triple_satisfies(a, b, c):
                        ok += 1
                    else:
                        triple_ok_all = 0.0
                triple_ok_fraction = ok / len(triples)

        combined = total if (valid_all and unique_support_ok == 1.0 and triple_ok_all == 1.0) else 0.0

        return {
            "size": total,
            "valid_fraction": float(valid_fraction),
            "unique_support_ok": float(unique_support_ok),
            "triple_ok_fraction": float(triple_ok_fraction),
            "combined_score": float(combined),
        }

    A_run, err = run_module(capture=False)
    if err is not None:
        return {"error": 0.0, "exec_error": err, "combined_score": 0.0}
    return _score_from_A(A_run)

if __name__ == "__main__":
    import json
    # Evaluate the sibling generator `initial_program.py`, not this evaluator file.
    # This script is intended to be run from the examples/admissible_set_7_5 directory
    # or from the repository root; compute the path relative to this file.
    here = os.path.dirname(__file__)
    program_path = os.path.join(here, "initial_program.py")
    if not os.path.exists(program_path):
        # fallback: try to find module by import if file not present
        try:
            from initial_program import generate_A  # type: ignore
            A = generate_A(N, W)
            print(json.dumps(_score_from_A(A), indent=2))
        except Exception as e:
            print(json.dumps({"error": "cannot locate initial_program.py", "exc": str(e)}))
    else:
        print(json.dumps(evaluate(program_path), indent=2))
