"""
Utility helpers for the cap set example.

Provides:
 - get_all_elements(n): all vectors in {0,1,2}^n as a NumPy array
 - is_capset(S, n): cap-set validity check (no nontrivial x+y+z=0 mod 3)
 - can_be_added(el, capset): greedy-friendly helper to keep the set valid
"""

import itertools
import numpy as np


def _encode_base3(vec: tuple[int, ...]) -> int:
    """Encode a base-3 vector to a single integer for fast set lookups."""
    val = 0
    mul = 1
    for v in vec:
        val += int(v) * mul
        mul *= 3
    return val


def _neg_sum_int(a_int: int, b_int: int, n: int) -> int:
    """
    Compute c_int such that a + b + c = 0 (mod 3) in encoded form.
    This is c = -a - b (mod 3) done digitwise in base 3.
    """
    res = 0
    mul = 1
    for _ in range(n):
        a_digit = a_int % 3
        b_digit = b_int % 3
        c_digit = (-(a_digit + b_digit)) % 3
        res += c_digit * mul
        a_int //= 3
        b_int //= 3
        mul *= 3
    return res


def get_all_elements(n: int) -> np.ndarray:
    """Return all vectors in {0,1,2}^n as a NumPy array of shape (3^n, n)."""
    return np.array(list(itertools.product([0, 1, 2], repeat=n)), dtype=int)


def is_capset(candidate_set, n: int) -> bool:
    """
    Check if a set is a cap set: no three distinct elements satisfy x + y + z = 0 (mod 3).
    """
    # normalize to tuples
    tuples = [tuple(int(x) for x in el) for el in candidate_set]

    # duplicate check
    if len(tuples) != len(set(tuples)):
        return False

    enc = [_encode_base3(t) for t in tuples]
    enc_set = set(enc)
    m = len(enc)

    for i in range(m):
        for j in range(i + 1, m):
            c_int = _neg_sum_int(enc[i], enc[j], n)
            if c_int in enc_set and c_int != enc[i] and c_int != enc[j]:
                return False
    return True


def can_be_added(element, capset) -> bool:
    """
    Return True if `element` can be appended to `capset` while keeping it a cap set.

    Complexity: O(|S|) using encoded lookup instead of rechecking the whole set.
    """
    n = len(element)
    elem_tuple = tuple(int(x) for x in element)
    elem_enc = _encode_base3(elem_tuple)

    # Encode existing set and guard against duplicates/length mismatch.
    enc_set = set()
    enc_list = []
    for el in capset:
        t = tuple(int(x) for x in el)
        if len(t) != n:
            return False  # invalid dimension mismatch
        enc_val = _encode_base3(t)
        if enc_val in enc_set:
            return False  # duplicate already breaks capset
        enc_set.add(enc_val)
        enc_list.append(enc_val)

    # Check if there exists a, b in capset with a + b + element = 0 (mod 3).
    for a_int in enc_list:
        b_int = _neg_sum_int(a_int, elem_enc, n)
        if b_int in enc_set and b_int != a_int:
            return False

    return True
