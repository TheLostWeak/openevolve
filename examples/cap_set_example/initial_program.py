import itertools
import random
from typing import Tuple, List, Callable

# Fixed random seed for reproducibility. LLMs may modify this seed or introduce randomness.
_RANDOM_SEED = 42
random.seed(_RANDOM_SEED)

def can_add(new_vec: Tuple[int, ...], existing_set: List[Tuple[int, ...]]) -> bool:
    """
    Check whether adding `new_vec` to `existing_set` preserves the Cap Set property.
    This performs an incremental validity check for efficiency.
    """
    existing = set(existing_set)
    
    # Check 1: Check for duplicate element (new_vec already in the set).
    if new_vec in existing:
        return False

    # Check 2: Check if `new_vec` forms an illegal triple with two vectors
    # from `existing_set`, i.e. whether there exist x, y in existing_set such
    # that x + y + new_vec == 0 (mod 3).
    for x in existing_set:
        y = tuple((-(x[i] + new_vec[i])) % 3 for i in range(len(new_vec)))
        if y in existing and y != x:
            return False

    return True

def _initial_priority(vec: Tuple[int, ...], n: int) -> float:
    """
    Initial priority function (the part subject to evolution).

    Args:
        vec: a vector in F_3^n (e.g. (0,1,0,2)).
        n: dimensionality of the vector space.

    Returns:
        A float score; higher scores are considered earlier by the greedy
        construction.

    Current implementation returns a random score and serves as a weak
    baseline. The LLM is expected to modify this function to discover
    priority rules that yield larger cap sets.
    """
    # Basic Strategy: fully random baseline (evolution starts from here)
    return random.random()

def gen_set(n: int, priority_func: Callable[[Tuple[int, ...], int], float] = None) -> List[Tuple[int, ...]]:
    """
    Construct a Cap Set in F_3^n using a greedy algorithm.

    Algorithm: generate all candidate vectors, sort them by `priority_func`,
    then iterate and add a vector if it keeps the set a valid Cap Set.

    Args:
        n: dimensionality.
        priority_func: priority function; if None the internal
          `_initial_priority` is used.

    Returns:
        A list of tuples representing a Cap Set (each tuple's coordinates are
        integers in {0,1,2}).
    """
    if priority_func is None:
        priority_func = _initial_priority
    
    # 1. Generate all vectors in F_3^n
    all_vectors = list(itertools.product([0, 1, 2], repeat=n))
    
    # 2. Sort candidates by priority (higher priority first)
    sorted_vectors = sorted(all_vectors,
                            key=lambda v: priority_func(v, n),
                            reverse=True)
    
    # 3. Greedy construction
    capset = []
    for vec in sorted_vectors:
        if can_add(vec, capset):
            capset.append(vec)
    
    return capset

def generate_set_for_eval(n: int) -> List[Tuple[int, ...]]:
    """
    Wrapper used by the evaluator. Uses the default initial priority function.

    Note: during evolution OpenEvolve/LLMs will rewrite parts of this module,
    so the wrapped priority function may be replaced by an evolved version.
    """
    return gen_set(n, priority_func=_initial_priority)

priority = _initial_priority
generate_set = generate_set_for_eval