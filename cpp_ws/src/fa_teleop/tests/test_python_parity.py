#!/usr/bin/env python3
"""Run copied original frame-construction methods without starting any components."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np

source, binary = sys.argv[1:3]
module = ast.parse(Path(source).read_text())
methods = []
for definition in module.body:
    if isinstance(definition, ast.ClassDef):
        methods += [node for node in definition.body if isinstance(node, ast.FunctionDef) and
                    node.name in {"_orthogonalize_frame", "_get_stable_hand_dir_frame"}]
assert len(methods) == 2
cls = ast.ClassDef(name="Reference", bases=[], keywords=[], body=methods, decorator_list=[])
code = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))

def normalize(value):
    n = np.linalg.norm(value)
    return value/n if np.isfinite(n) and n >= 1e-8 else None

context = {"np": np, "_safe_normalize": normalize, "robots": SimpleNamespace(LEFT="left")}
exec(compile(code, "copied_original_frame_methods", "exec"), context)
mapping = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
for sample in json.loads(subprocess.check_output([binary], text=True)):
    obj = context["Reference"]()
    obj.hand_side = "left" if sample["left"] else "right"
    obj.wrist_idx, obj.index_knuckle_idx, obj.middle_knuckle_idx, obj.pinky_knuckle_idx = 0, 7, 12, 22
    frame = obj._get_stable_hand_dir_frame(np.array(sample["points"]))
    expected = np.array(frame[1:]).T
    np.testing.assert_allclose(sample["R"], expected, atol=1e-12, rtol=0)
    target_R = mapping @ (expected @ np.array(sample["baseline_R"]).T) @ mapping.T @ np.array(sample["robot_R"])
    target_p = np.array(sample["robot_p"]) + mapping @ (np.array(frame[0]) - sample["baseline_p"])
    np.testing.assert_allclose(sample["target_R"], target_R, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sample["target_p"], target_p, atol=1e-12, rtol=0)
print("PASS both hand frames and base-frame retargeting against copied Python behavior at 1e-12 tolerance")
