from typing import List, Tuple, Dict
import itertools
import os
import json

TRIPLES = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
    (0, 1, 2),
    (0, 2, 1),
    (1, 1, 1),
    (2, 2, 2),
]
INT_TO_WEIGHT = [0, 1, 1, 2, 2, 3, 3]


def expand_admissible_set(pre_admissible_set: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Expand group-pattern rows into full vectors using cyclic rotations."""
    expanded = []
    for row in pre_admissible_set:
        rlists = []
        for pid in row:
            x, y, z = TRIPLES[int(pid)]
            r = [(x, y, z)]
            if not (x == y == z):
                r.append((z, x, y))
                r.append((y, z, x))
            rlists.append(r)
        for prod in itertools.product(*rlists):
            expanded.append(sum(prod, ()))
    return expanded


# Precompute compatibility: for each unordered pair (p1,p2) and group length g_len (1..3),
# compute set of p3 values such that there exists rotations/position where (p1,p2,p3)
# produce an allowed multiset at some internal coordinate.
COMPAT_PAIR_TO_P3 = {}
_ALLOWED = ({0, 0, 1}, {0, 0, 2}, {0, 1, 2})
for g_len in (1, 2, 3):
    for p1 in range(len(TRIPLES)):
        for p2 in range(len(TRIPLES)):
            key = (min(p1, p2), max(p1, p2), g_len)
            s = set()
            pat1 = TRIPLES[p1]
            pat2 = TRIPLES[p2]
            rots1 = [pat1]
            rots2 = [pat2]
            if not (pat1[0] == pat1[1] == pat1[2]):
                rots1 = [pat1, (pat1[2], pat1[0], pat1[1]), (pat1[1], pat1[2], pat1[0])]
            if not (pat2[0] == pat2[1] == pat2[2]):
                rots2 = [pat2, (pat2[2], pat2[0], pat2[1]), (pat2[1], pat2[2], pat2[0])]
            for p3 in range(len(TRIPLES)):
                pat3 = TRIPLES[p3]
                rots3 = [pat3]
                if not (pat3[0] == pat3[1] == pat3[2]):
                    rots3 = [pat3, (pat3[2], pat3[0], pat3[1]), (pat3[1], pat3[2], pat3[0])]
                found = False
                for a in rots1:
                    for b in rots2:
                        for c in rots3:
                            for pos in range(g_len):
                                vals = tuple(sorted((a[pos], b[pos], c[pos])))
                                if vals == (0, 0, 1) or vals == (0, 0, 2) or vals == (0, 1, 2):
                                    s.add(p3)
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
            COMPAT_PAIR_TO_P3[key] = s


def priority(el: Tuple[int, ...]) -> float:
    """Deterministic priority for a full vector `el`.

    Heuristic: favor vectors with more 2s, then more 1s, and then larger
    position-weighted sum to break ties. This is deterministic and cheap.
    """
    ones = el.count(1)
    twos = el.count(2)
    pos_sum = sum(i * v for i, v in enumerate(el))
    return twos * 3.0 + ones * 1.0 + 1e-6 * pos_sum


def generate_A(n: int = 12, w: int = 7, seed: int = 0) -> List[List[int]]:
    """Group-level greedy generator.

    Steps:
    1. Assert n divisible by 3 and compute `num_groups`.
    2. Enumerate all group-children (product over 7 patterns), keep those with total weight w.
    3. For each child compute a representative full vector and a priority score.
    4. Sort children by priority (desc) and greedily add a child if the expanded
       admissible set (after adding it) still satisfies the triple condition.
    5. Return the expanded admissible set (list of full vectors).
    """
    assert n % 3 == 0, "n must be multiple of 3 for grouped representation"
    num_groups = n // 3

    # enumerate valid children (group-pattern rows) with total weight w
    valid_children = []
    for child in itertools.product(range(7), repeat=num_groups):
        weight = sum(INT_TO_WEIGHT[int(p)] for p in child)
        if weight == w:
            valid_children.append(tuple(int(x) for x in child))

    # compute representative vector (first rotation) and priority for each child
    scored = []
    for child in valid_children:
        rep = []
        for pid in child:
            rep.extend(TRIPLES[pid])
        rep = tuple(rep[:n])
        scored.append((priority(rep), child, rep))

    scored.sort(key=lambda x: (-x[0], x[1]))

    # Load precomputed incompatible unordered pattern triples if available
    bad_unordered = set()
    try:
        compat_path = os.path.join(os.path.dirname(__file__), 'pattern_triple_compat.json')
        if os.path.exists(compat_path):
            with open(compat_path, 'r', encoding='utf-8') as f:
                j = json.load(f)
                for t in j.get('incompatible', []):
                    bad_unordered.add(tuple(sorted(t)))
    except Exception:
        bad_unordered = set()

    def triple_ok(a, b, c):
        for i in range(len(a)):
            vals = sorted([a[i], b[i], c[i]])
            if vals == [0, 0, 1] or vals == [0, 0, 2] or vals == [0, 1, 2]:
                return True
        return False

    # Greedy selection in group-space: maintain pre_admissible_set (rows)
    # Use purely group-level checks (no full expansion)
    pre_adm: List[Tuple[int, ...]] = []

    def triple_ok_groups(r1: Tuple[int, ...], r2: Tuple[int, ...], r3: Tuple[int, ...]) -> bool:
        """Return True iff the triple of group-pattern rows can produce
        at least one coordinate (in some group at some internal position)
        where the three values form an allowed multiset.

        This checks rotations per group without expanding full vectors.
        """
        group_size = 3
        num_groups = len(r1)
        # cache key for triple rows
        for gi in range(num_groups):
            p1 = int(r1[gi])
            p2 = int(r2[gi])
            p3 = int(r3[gi])
            n = num_groups * group_size
            start = gi * group_size
            end = min(start + group_size, n)
            g_len = end - start
            key = (min(p1, p2), max(p1, p2), g_len)
            compat_set = COMPAT_PAIR_TO_P3.get(key, set())
            if p3 in compat_set:
                return True
        return False

    from itertools import combinations_with_replacement

    # cache for triple results (unordered triple of rows -> bool)
    triple_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], bool] = {}

    for _, child, _ in scored:
        # fast conservative prune using precomputed unordered incompatibilities
        reject = False
        if bad_unordered and len(pre_adm) >= 2:
            for r1, r2 in itertools.combinations(pre_adm, 2):
                # quick check using pair->compatible-p3 mapping
                all_bad = True
                for gi in range(num_groups):
                    pair_key = (min(r1[gi], r2[gi]), max(r1[gi], r2[gi]), min(3, n - gi*3 if n - gi*3>0 else 3))
                    compat_set = COMPAT_PAIR_TO_P3.get(pair_key, set())
                    if child[gi] in compat_set:
                        all_bad = False
                        break
                if all_bad:
                    reject = True
                    break
        if reject:
            continue

        # Incremental triple checks: only check triples that include `child`.
        m = len(pre_adm)
        ok = True

        # helper to probe cache and compute if needed
        def probe_and_store(a_row, b_row, c_row):
            key = tuple(sorted((tuple(a_row), tuple(b_row), tuple(c_row))))
            if key in triple_cache:
                return triple_cache[key]
            res = triple_ok_groups(a_row, b_row, c_row)
            triple_cache[key] = res
            return res

        # triples (child, existing_i, existing_j) for i <= j
        for i in range(m):
            for j in range(i, m):
                if not probe_and_store(child, pre_adm[i], pre_adm[j]):
                    ok = False
                    break
            if not ok:
                break

        if not ok:
            continue

        # triples (child, child, existing_i)
        for i in range(m):
            if not probe_and_store(child, child, pre_adm[i]):
                ok = False
                break
        if not ok:
            continue

        # triple (child, child, child)
        if not probe_and_store(child, child, child):
            continue

        pre_adm.append(child)

    final_expanded = expand_admissible_set(pre_adm)
    return [list(v) for v in final_expanded]


if __name__ == "__main__":
    import json

    print(json.dumps(generate_A(12, 7)[:10], indent=2))

    if __name__ == "__main__":
        import json

        print(json.dumps(generate_A(12, 7)[:10], indent=2))
    import json

    # Example for (12,7): number of groups = ceil(12/3) = 4
    print(json.dumps(generate_A(12, 7), indent=2))
