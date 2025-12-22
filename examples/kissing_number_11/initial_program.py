"""
Naive algorithmic initial program for Kissing Number problem (d=11).

Core idea:
- Maintain a current point set C
- Repeatedly try to generate ONE new point
- Accept it only if global distance constraint is satisfied
"""

d = 11
M = 10**12


def squared_norm(x):
    return sum(v * v for v in x)


def squared_dist(x, y):
    return sum((a - b) * (a - b) for a, b in zip(x, y))


def try_add_point(C):
    """
    Attempt to generate ONE new point that can be added to C.

    Returns:
        - a valid new point (list of ints), or
        - None if no valid point is found by this naive strategy
    """

    # Naive candidate generation:
    # try a small fixed set of simple integer vectors
    candidates = []

    # single-axis candidates
    for i in range(d):
        v = [0] * d
        v[i] = M
        candidates.append(v)

        v_neg = [0] * d
        v_neg[i] = -M
        candidates.append(v_neg)

    # test candidates one by one
    for v in candidates:
        # exclude origin just in case
        if all(coord == 0 for coord in v):
            continue

        # compute max norm if v were added
        max_norm_sq = squared_norm(v)
        for x in C:
            max_norm_sq = max(max_norm_sq, squared_norm(x))

        # check global distance domination
        ok = True
        for x in C:
            if squared_dist(x, v) < max_norm_sq:
                ok = False
                break

        if ok:
            return v

    return None


def get_C():
    # EVOLVE-BLOCK-START

    # start with a single non-zero point
    C = [[M] + [0] * (d - 1)]

    # repeatedly try to add new points
    while True:
        v = try_add_point(C)
        if v is None:
            break
        C.append(v)

    return C

    # EVOLVE-BLOCK-END


if __name__ == "__main__":
    print("Initial C size:", len(get_C()))
