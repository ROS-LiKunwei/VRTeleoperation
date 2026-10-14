#!/usr/bin/python3
"""Record built binaries and resolved runtime dependencies after a successful build."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--prefix", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
prefix = args.prefix.resolve()
bundle = prefix / "share/fa_teleop/bundle"
release = json.loads((bundle / "release.json").read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
binaries, dependencies = {}, {}
for path in sorted((prefix / "lib/fa_teleop").iterdir()):
    if not path.is_file() or path.suffix == ".py":
        continue
    if path.read_bytes()[:4] != b"\x7fELF":
        continue
    binaries[str(path.relative_to(prefix))] = sha(path)
    result = subprocess.run(["ldd", str(path)], text=True, capture_output=True, check=True)
    for line in result.stdout.splitlines():
        fields = line.split()
        resolved = fields[2] if len(fields) >= 3 and fields[1] == "=>" else (fields[0] if fields else "")
        lib = Path(resolved)
        if lib.is_absolute() and lib.is_file():
            dependencies[str(lib)] = sha(lib)
report = {"schema": 1, "release_id": sha(bundle / "release.json"), "components": release["components"],
          "platform": platform.platform(), "machine": platform.machine(), "binaries": binaries,
          "runtime_dependencies_sha256": dependencies,
          "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0]}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print("Build receipt:", args.output)
