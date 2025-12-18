"""
Initial program for Kissing Number problem (d=11).

This module should provide a function `get_C()` that returns a list of points,
each point is an iterable of `d` numeric coordinates. OpenEvolve will evolve
the code inside the EVOLVE block.

Interface expected by evaluator:
  - get_C() -> List[List[float]]

Start with a small example set; the LLM will modify this during evolution.
"""

d = 11

def get_C():
    # EVOLVE-BLOCK-START
    # A small initial packing in R^11 (two simple orthogonal unit vectors)
    # Use integer coordinates as required by evaluator
    return [
        [1000000000000] + [0] * (d - 1),
        [0, 1000000000000] + [0] * (d - 2),
    ]
    # EVOLVE-BLOCK-END


if __name__ == "__main__":
    print("Initial C size:", len(get_C()))
