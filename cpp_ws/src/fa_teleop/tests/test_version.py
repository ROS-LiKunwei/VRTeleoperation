#!/usr/bin/env python3
import pathlib
import shutil
import subprocess
import sys
import tempfile

binary, bundle = sys.argv[1:3]
with tempfile.TemporaryDirectory(prefix="fa-version-") as temporary:
    root = pathlib.Path(temporary) / "bundle"
    shutil.copytree(bundle, root)
    subprocess.run([binary, str(root), "--verify-only"], check=True)
    for name in ["vendor/ik_7dof/src/fa_ik_solver.cpp", "vendor/min_snap/src/min_snap_trajectory.cpp",
                 "models/fa_description/urdf/fa_robot.urdf", "config/precision_200hz.yaml", "release.json", "files.sha256"]:
        path = root / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n# tamper\n")
        result = subprocess.run([binary, str(root), "--verify-only"], capture_output=True, text=True)
        assert result.returncode != 0, name
        path.write_bytes(original)
    subprocess.run([binary, str(root), "--verify-only"], check=True)
print("PASS relocation and six independent version-mismatch rejections")
