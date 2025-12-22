"""SOTA solution for hex_packing example.

This module exposes a top-level variable `solution` which is a flat list
of 12*(x,y,angle) floats (same format as `initial_program.py`) so it can
be consumed directly by the evaluator.
"""

# Inner hexagon parameters (x, y, angle_degrees) for 12 hexagons
inner_hex_data = [
    [-4.189650776376798, 1.8191996226926388, 27.083656548467786],
    [-4.1762891342159225, 7.064090887028092, 267.1521094125118],
    [-2.742942219672671, 3.9055564132509595, 27.098852933212644],
    [-6.194948523724643, 4.243623380441444, 87.15308054130165],
    [-2.4735773559547236, 6.43056712077689, 27.144807633484653],
    [-3.8735033091694633, 5.272827562035259, 207.12012595575163],
    [-1.040517842157661, 3.2724339122261963, 147.11193087313353],
    [-1.3431542020437193, 5.063486444235776, 27.124004797633685],
    [-5.89197412464008, 2.452466762507662, 207.10748153590382],
    [-4.492206360794807, 3.610020660157929, 327.09918513988094],
    [-2.4402287984241076, 2.1146830723057635, 147.1073847274004],
    [-5.576327284388157, 5.906127413424367, 207.16788311720808],
]

# Flatten into the evaluator's expected `solution` format: [x,y,angle, x,y,angle, ...]
solution = []
for trip in inner_hex_data:
    solution.extend([float(trip[0]), float(trip[1]), float(trip[2])])


if __name__ == "__main__":
    import json

    print(json.dumps(solution))

# Known SOTA outer hexagon parameters (from original data)
outer_hex_center = (-3.7029107170331157, 4.262832492467158)
outer_hex_side_length = 3.9419123
outer_hex_angle_degrees = 219.59113156095745