#!/usr/bin/python3
"""Override only a temporary MuJoCo copy and verify startup rejects its hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument("--prefix", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
prefix = args.prefix.resolve()
library = next((prefix / "lib/fa_teleop/runtime").glob("libmujoco.so.*"))
with tempfile.TemporaryDirectory(prefix="fa-mujoco-lock-") as temporary:
    copy = Path(temporary) / library.name
    shutil.copyfile(library, copy)
    with copy.open("ab") as stream:
        stream.write(b"\nFA runtime version mismatch test\n")
    environment = dict(os.environ, ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="194")
    environment["LD_LIBRARY_PATH"] = temporary + ":" + environment.get("LD_LIBRARY_PATH", "")
    result = subprocess.run([str(prefix / "lib/fa_teleop/fa_mujoco_node"), "--ros-args", "-r", "__ns:=/fa_runtime_lock_test", "-p", "control_backend:=mujoco"],
                            env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "loaded runtime SHA" in result.stderr, result.stdout + result.stderr
args.output.parent.mkdir(parents=True, exist_ok=True)
release = hashlib.sha256((prefix / "share/fa_teleop/bundle/release.json").read_bytes()).hexdigest()
args.output.write_text(json.dumps({"passed": True, "release_id": release, "temporary_runtime_override_rejected": True,
                                  "original_runtime_modified": False, "physical_hardware": False}, indent=2)+"\n")
print("PASS mismatched MuJoCo runtime rejected before model/feedback startup")
