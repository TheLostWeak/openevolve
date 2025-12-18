"""
Evaluator for the Kissing Number problem in R^d (d=11).

Scoring rule (hard constraint):
  - If C satisfies: 0 not in C and min_{x!=y in C} ||x-y|| >= max_{x in C} ||x||,
    then score = |C| (size of set), otherwise score = 0.

The evaluator looks for a `get_C()` function or a variable `C` in the evolved
program module. It returns a dict containing `combined_score` and diagnostic
metrics that can be used as features by MAP-Elites.
"""

from math import sqrt
import importlib.util
import traceback
from typing import List, Tuple, Any


def _to_point(p: Any, d: int) -> Tuple[int, ...]:
    for x in p:
        if not isinstance(x, int):
            raise ValueError("Coordinates must be integers")
    
    coords = tuple(int(x) for x in p)
    
    if len(coords) != d:
        raise ValueError(f"Point must have dimension {d}, got {len(coords)}")

    return coords


def norm2(p: Tuple[int, ...]):
    s = 0
    for x in p:
        s += x * x
    return s


def pairwise_min_distance2(points: List[Tuple[int, ...]]) -> int:
    n = len(points)
    if n < 2:
        return -1  # undefined for <2 points
    d = len(points[0]) if n > 0 else 0
    min_d = norm2(tuple(points[0][k] - points[1][k] for k in range(d)))
    for i in range(n):
        for j in range(i + 1, n):
            min_d = min(min_d, norm2(tuple(points[i][k] - points[j][k] for k in range(d))))
    return min_d


def _load_module(path: str, name: str = "evolved"):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load module from path: " + path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(program_path: str) -> dict:
    """Evaluate the evolved program file at `program_path`.

    Returns a dict with `combined_score` (the size if valid, otherwise 0)
    and diagnostic metrics: `size`, `max_norm`, `min_pairwise`, and optionally `error`.
    """
    d = 11
    try:
        module = _load_module(program_path)

        # Find candidate set: prefer get_C(), else variable C
        if hasattr(module, "get_C") and callable(getattr(module, "get_C")):
            raw = module.get_C()
        elif hasattr(module, "C"):
            raw = getattr(module, "C")
        else:
            return {
                "combined_score": 0.0,
                "size": 0,
                "max_norm": 0.0,
                "min_pairwise": 0.0,
                "error": "No get_C() or C found",
            }

        # Materialize points and validate
        points = []
        for p in raw:
            pt = _to_point(p, d)
            points.append(pt)
        
        size = len(points)
            
        if size == 0:
            return {
                "combined_score": 0.0,
                "size": 0,
                "max_norm": 0.0,
                "min_pairwise": 0.0,
                "error": "Empty set C",
            }
        
        n2 = [norm2(p) for p in points]
        max_norm2 = max(n2)
        
        if size == 1:
            if max_norm2 == 0:
                return {
                    "combined_score": 0.0,
                    "size": 1,
                    "max_norm": 0.0,
                    "min_pairwise": float('inf'),
                    "error": "Zero vector present in C",
                }
            else:
                return {
                    "combined_score": 1.0,
                    "size": 1,
                    "max_norm": sqrt(max_norm2),
                    "min_pairwise": float('inf'),
                }
        
        min_pair2 = pairwise_min_distance2(points)    
            
        if any(n == 0 for n in n2):
            return {
                "combined_score": 0.0,
                "size": len(points),
                "max_norm": sqrt(max_norm2),
                "min_pairwise": sqrt(min_pair2),
                "error": "Zero vector present in C",
            }
        
        if min_pair2 >= max_norm2:
            return {
                "combined_score": float(size),
                "size": size,
                "max_norm": sqrt(max_norm2),
                "min_pairwise": sqrt(min_pair2),
            }
        else:
            return {
                "combined_score": 0.0,
                "size": size,
                "max_norm": sqrt(max_norm2),
                "min_pairwise": sqrt(min_pair2),
                "error": "Constraint violated: min_pairwise < max_norm",
            }

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "combined_score": 0.0,
            "size": 0,
            "max_norm": 0.0,
            "min_pairwise": 0.0,
            "error": str(e),
            "traceback": tb,
        }


if __name__ == "__main__":
    import sys
    res = evaluate(sys.argv[1]) if len(sys.argv) > 1 else {"error": "No path"}
    print(res)
