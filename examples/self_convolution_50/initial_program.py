"""
Initial program for the self-convolution constant search problem.

We consider nonnegative step functions supported on [-1/4, 1/4] with 50 equal-width
intervals. The EVOLVE block below should contain a Python list named `steps`
with 50 nonnegative float values (heights of each step). The evaluator will
construct the piecewise-constant function and compute the ratio

    R(f) = ||f*f||_2^2 / (||f*f||_1 * ||f*f||_infty)

OpenEvolve will attempt to maximize `R(f)`; the minimal universal constant C
must be at least the supremum of R(f).
"""

# Number of equal-width bins on [-1/4, 1/4]
NUM_BINS = 50

# EVOLVE-BLOCK-START
steps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
         11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
         21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
         31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
         41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0]

def get_steps():
    """Return the list of 50 nonnegative step heights."""
    return steps
# EVOLVE-BLOCK-END