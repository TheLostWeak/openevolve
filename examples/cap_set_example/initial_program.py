"""
Minimal cap set constructor template for F_3^n.

This module intentionally contains NO domain-specific heuristics.
It provides only:
- a correctness-preserving incremental construction loop
- generic, weakly-typed tunable parameters
- deterministic randomness

Designed as an evolution starting point.
"""

import random
from typing import List, Tuple

_RANDOM_SEED = 0


# ---------- Basic arithmetic ----------

def add_mod3(a, b):
    return tuple((x + y) % 3 for x, y in zip(a, b))


def neg_mod3(a):
    return tuple((-x) % 3 for x in a)


def third_point(a, b):
    return neg_mod3(add_mod3(a, b))


# ---------- Validity check ----------

def is_valid_extension(S, p):
    for x in S:
        if third_point(x, p) in S:
            return False
    return True


# ---------- Builder ----------

class CapSetBuilder:
    def __init__(
        self,
        *,
        max_steps: int = 5000,
        sample_pool: int = 200,
        accept_prob: float = 1.0,
        seed: int = _RANDOM_SEED,
    ):
        self.max_steps = max_steps
        self.sample_pool = sample_pool
        self.accept_prob = accept_prob
        self.rng = random.Random(seed)

    def build(self, n: int) -> List[Tuple[int, ...]]:
        points = [tuple(self.rng.randrange(3) for _ in range(n))
                  for _ in range(3 ** n)]

        S = []
        seen = set()

        for _ in range(self.max_steps):
            if not points:
                break

            # Weak candidate sampling
            cand = self.rng.sample(points, min(self.sample_pool, len(points)))

            p = cand[self.rng.randrange(len(cand))]

            if p in seen:
                continue

            if is_valid_extension(S, p):
                if self.rng.random() <= self.accept_prob:
                    S.append(p)
                    seen.add(p)

        return S


# ---------- Public API ----------

def generate_set(n: int) -> List[Tuple[int, ...]]:
    if n <= 0:
        return []

    builder = CapSetBuilder()
    return builder.build(n)


if __name__ == "__main__":
    A = generate_set(8)
    print(len(A))
