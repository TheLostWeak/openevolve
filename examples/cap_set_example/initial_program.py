import itertools
import random
from typing import Tuple, List, Callable, Optional

# Hard-coded dimension for this example. All generator code should use this value.
CAP_N = 6

# Fixed random seed for reproducibility. LLMs may modify this seed or introduce randomness.
_RANDOM_SEED = 42
random.seed(_RANDOM_SEED)

def can_add(new_vec: Tuple[int, ...], existing_list: List[Tuple[int, ...]], existing_set: Optional[set] = None) -> bool:
    """
    Efficient incremental check for whether `new_vec` can be added to the
    current cap set without violating the cap set property.

    This function accepts both the list of existing vectors (in insertion
    order) and an optional set for O(1) membership checks. If `existing_set`
    is not provided it will be constructed from `existing_list`.

    Args:
        new_vec: Candidate vector to add.
        existing_list: List of vectors already in the cap set (in order).
        existing_set: Optional set of those vectors for O(1) lookups.

    Returns:
        True if `new_vec` can be added without creating a 3-term arithmetic
        progression (mod 3), otherwise False.
    """
    n = len(new_vec)
    if existing_set is None:
        existing_set = set(existing_list)

    # Duplicate check
    if new_vec in existing_set:
        return False
    # For each existing vector x compute y = -(x + new_vec) and check if y
    # is present in the set. This is O(m) checks with O(1) membership tests.
    for x in existing_list:
        # Build the tuple directly; zip is slightly faster and clearer.
        y = tuple((-(a + b)) % 3 for a, b in zip(x, new_vec))
        if y in existing_set and y != x:
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
    sorted_vectors = sorted(all_vectors, key=lambda v: priority_func(v, n), reverse=True)

    # 3. Greedy construction with a persistent set for O(1) membership checks
    capset: List[Tuple[int, ...]] = []
    capset_set = set()

    for vec in sorted_vectors:
        # fast skip duplicates
        if vec in capset_set:
            continue

        if can_add(vec, capset, capset_set):
            capset.append(vec)
            capset_set.add(vec)

    return capset

def generate_set_for_eval(n: int) -> List[Tuple[int, ...]]:
    """
    Wrapper used by the evaluator. For this example `n` is fixed to `CAP_N`.
    """
    return gen_set(CAP_N, priority_func=_initial_priority)


priority = _initial_priority
generate_set = generate_set_for_eval