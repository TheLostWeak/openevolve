"""Hexagon packing evaluator for n regular hexagons (unit side).

Provides `evaluate(program_text_or_path)` which accepts either a path to a
program file or the program text. The program should expose a list/array
named `solution` or contain a JSON array of floats. The expected format is
an array of length `n*3` where each triple is (x, y, angle_degrees) for a
unit-side regular hexagon.

Returns a dict with metric `side_length` (smaller is better). Invalid
solutions (overlap or parse errors) are penalized with a large side_length.
"""

from __future__ import annotations

import math
import json
import os
from typing import List, Tuple, Any
import re
import ast

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:
    EvaluationResult = None


N = 12
PENALTY = 1e6


def hexagon_vertices(center_x: float, center_y: float, side_length: float, angle_degrees: float) -> List[Tuple[float, float]]:
    vertices = []
    angle_radians = math.radians(angle_degrees)
    for i in range(6):
        angle = angle_radians + 2 * math.pi * i / 6
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices


def normalize_vector(v: Tuple[float, float]) -> Tuple[float, float]:
    mag = math.hypot(v[0], v[1])
    return (v[0] / mag, v[1] / mag) if mag != 0 else (0.0, 0.0)


def get_normals(vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    normals = []
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = normalize_vector((-edge[1], edge[0]))
        normals.append(normal)
    return normals


def project_polygon(vertices: List[Tuple[float, float]], axis: Tuple[float, float]) -> Tuple[float, float]:
    min_p = float("inf")
    max_p = float("-inf")
    for vx, vy in vertices:
        proj = vx * axis[0] + vy * axis[1]
        if proj < min_p:
            min_p = proj
        if proj > max_p:
            max_p = proj
    return min_p, max_p


def overlap_1d(min1: float, max1: float, min2: float, max2: float) -> bool:
    return max1 >= min2 and max2 >= min1


def polygons_intersect(vertices1: List[Tuple[float, float]], vertices2: List[Tuple[float, float]]) -> bool:
    normals1 = get_normals(vertices1)
    normals2 = get_normals(vertices2)
    axes = normals1 + normals2
    for axis in axes:
        min1, max1 = project_polygon(vertices1, axis)
        min2, max2 = project_polygon(vertices2, axis)
        if not overlap_1d(min1, max1, min2, max2):
            return False
    return True


def hexagons_are_disjoint(hex1_params: Tuple[float, float, float, float], hex2_params: Tuple[float, float, float, float]) -> bool:
    v1 = hexagon_vertices(*hex1_params)
    v2 = hexagon_vertices(*hex2_params)
    return not polygons_intersect(v1, v2)


def is_inside_hexagon(point: Tuple[float, float], hex_params: Tuple[float, float, float, float]) -> bool:
    vertices = hexagon_vertices(*hex_params)
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        pv = (point[0] - p1[0], point[1] - p1[1])
        cross = edge[0] * pv[1] - edge[1] * pv[0]
        if cross < -1e-9:
            return False
    return True


def all_points_contained(points: List[Tuple[float, float]], center: Tuple[float, float], side_length: float, angle_deg: float) -> bool:
    outer = (center[0], center[1], side_length, angle_deg)
    for p in points:
        if not is_inside_hexagon(p, outer):
            return False
    return True


def parse_solution(text: str) -> List[float]:
    # 1) If JSON array exactly
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [float(x) for x in obj]
    except Exception:
        pass

    # 2) If text contains a fenced code block, extract its contents
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    code = m.group(1) if m else text

    # 3) Try AST parsing to find a top-level assignment to `solution`
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "solution":
                        try:
                            lit = ast.literal_eval(node.value)
                            if isinstance(lit, (list, tuple)):
                                return [float(x) for x in lit]
                        except Exception:
                            pass
        # Also handle the case where the whole code is an expression that is a list
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
            try:
                lit = ast.literal_eval(tree.body[0].value)
                if isinstance(lit, (list, tuple)):
                    return [float(x) for x in lit]
            except Exception:
                pass
    except Exception:
        pass

    # 4) Fallback: try eval of the code (last resort)
    try:
        val = eval(code, {}, {})
        if isinstance(val, (list, tuple)):
            return [float(x) for x in val]
    except Exception:
        pass

    raise ValueError("Unable to parse solution vector from program text")


def evaluate(program_text_or_path: str) -> dict:
    """Main entry used by the framework.

    Returns a dict with `side_length` (float). Smaller is better.
    Invalid solutions return a large penalized side_length.
    """
    # Load text
    if os.path.exists(program_text_or_path):
        with open(program_text_or_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = program_text_or_path

    try:
        vec = parse_solution(text)
    except Exception as e:
        msg = f"parse_error: {e}"
        if EvaluationResult is not None:
            return EvaluationResult(metrics={"side_length": PENALTY, "combined_score": 0.0}, artifacts={"error": msg})
        return {"side_length": PENALTY, "combined_score": 0.0}

    if len(vec) != N * 3:
        msg = f"expected {N*3} floats, got {len(vec)}"
        if EvaluationResult is not None:
            return EvaluationResult(metrics={"side_length": PENALTY, "combined_score": 0.0}, artifacts={"error": msg})
        return {"side_length": PENALTY, "combined_score": 0.0}

    # Build inner hex params and vertices
    inner_params = []
    all_points: List[Tuple[float, float]] = []
    for i in range(N):
        x = vec[3 * i]
        y = vec[3 * i + 1]
        angle = vec[3 * i + 2]
        inner_params.append((x, y, 1.0, angle))
        all_points.extend(hexagon_vertices(x, y, 1.0, angle))

    # Check pairwise disjointness
    for i in range(N):
        for j in range(i + 1, N):
            if not hexagons_are_disjoint(inner_params[i], inner_params[j]):
                msg = f"overlap between {i} and {j}"
                if EvaluationResult is not None:
                    return EvaluationResult(metrics={"side_length": PENALTY, "combined_score": 0.0}, artifacts={"error": msg})
                return {"side_length": PENALTY, "combined_score": 0.0}

    # REQUIRE: programs must provide outer hexagon parameters. Evaluator will
    # only validate provided parameters and will not compute the minimal
    # enclosing hexagon automatically.
    ns: dict[str, Any] = {}
    try:
        exec(text, {}, ns)
    except Exception:
        ns = {}

    # Prefer AST extraction of top-level values to avoid running arbitrary code
    def _extract_top_level(name: str):
        # check fenced code first
        m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
        code = m.group(1) if m else text
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == name:
                            try:
                                return ast.literal_eval(node.value)
                            except Exception:
                                return None
        except Exception:
            return None
        return None

    center_val = _extract_top_level("outer_hex_center")
    side_val = _extract_top_level("outer_hex_side_length")
    angle_val = _extract_top_level("outer_hex_angle_degrees")

    if center_val is None or side_val is None or angle_val is None:
        msg = "program must define outer_hex_center, outer_hex_side_length, and outer_hex_angle_degrees"
        if EvaluationResult is not None:
            return EvaluationResult(metrics={"side_length": PENALTY, "combined_score": 0.0}, artifacts={"error": msg})
        return {"side_length": PENALTY, "combined_score": 0.0}

    try:
        center = (float(center_val[0]), float(center_val[1]))
        side_length = float(side_val)
        outer_angle = float(angle_val)
    except Exception as e:
        msg = f"invalid outer hex params: {e}"
        if EvaluationResult is not None:
            return EvaluationResult(metrics={"side_length": PENALTY, "combined_score": 0.0}, artifacts={"error": msg})
        return {"side_length": PENALTY, "combined_score": 0.0}

    # Validate containment
    if not all_points_contained(all_points, center, side_length, outer_angle):
        msg = "provided outer hexagon does not contain all inner vertices"
        if EvaluationResult is not None:
            return EvaluationResult(metrics={"side_length": PENALTY, "combined_score": 0.0}, artifacts={"error": msg})
        return {"side_length": PENALTY, "combined_score": 0.0}

    # Map combined score
    if side_length >= PENALTY or not math.isfinite(side_length):
        combined_score = 0.0
    else:
        combined_score = max(0.0, float(N) - side_length)

    # Use module-level EvaluationResult when available to avoid local-name shadowing
    if EvaluationResult is not None:
        return EvaluationResult(
            metrics={"side_length": float(side_length), "combined_score": combined_score},
            artifacts={"n_hexagons": N, "used_outer": True, "best_center": center},
        )
    return {"side_length": float(side_length), "combined_score": combined_score, "used_outer": True, "best_center": center}


if __name__ == "__main__":
    # Example: place 12 hex centers on a circle
    sol = []
    for i in range(N):
        theta = 2 * math.pi * i / N
        sol.extend([2.5 * math.cos(theta), 2.5 * math.sin(theta), 0.0])
    print("Example solution side_length:")
    print(evaluate(json.dumps(sol)))
