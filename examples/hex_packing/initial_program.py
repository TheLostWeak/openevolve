"""Initial program for hex packing example.

Defines `solution` as a Python list of 12*(x,y,angle) triples. Kept inside
the `hex_packing` folder as requested.
"""

solution = []
spacing = 2.2
rows = 3
cols = 4
cnt = 0
for r in range(rows):
    for c in range(cols):
        if cnt >= 12:
            break
        x = (c - (cols - 1) / 2.0) * spacing
        y = (r - (rows - 1) / 2.0) * spacing
        solution.extend([x, y, 0.0])
        cnt += 1

def _hexagon_vertices(center_x: float, center_y: float, side_length: float = 1.0, angle_degrees: float = 0.0):
    import math
    verts = []
    ang = math.radians(angle_degrees)
    for i in range(6):
        a = ang + 2 * math.pi * i / 6
        verts.append((center_x + side_length * math.cos(a), center_y + side_length * math.sin(a)))
    return verts
# Compute a conservative outer hexagon estimate at module load time so the
# evaluator (which executes this module with `exec`) can read these variables.
all_points = []
for i in range(0, len(solution), 3):
    cx = solution[i]
    cy = solution[i + 1]
    ang = solution[i + 2]
    all_points.extend(_hexagon_vertices(cx, cy, 1.0, ang))

import math

_centroid_x = sum(p[0] for p in all_points) / len(all_points)
_centroid_y = sum(p[1] for p in all_points) / len(all_points)
max_r = max(math.hypot(px - _centroid_x, py - _centroid_y) for px, py in all_points)

# Conservative outer hexagon provided by the starter program — programs are
# expected to provide `outer_hex_center`, `outer_hex_side_length`, and
# `outer_hex_angle_degrees` for the evaluator to validate.
outer_hex_center = (_centroid_x, _centroid_y)
outer_hex_side_length = max_r + 1.0
outer_hex_angle_degrees = 0.0


if __name__ == '__main__':
    import json

    # Print the solution (tools expect this)
    print(json.dumps(solution))
