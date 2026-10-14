#!/usr/bin/python3
"""Read-only verification that original protected source files did not change."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("snapshot", type=Path)
args = p.parse_args()
manifest = json.loads(args.snapshot.read_text())
root = Path(manifest["source_root"])
failures = []
for name, digest in manifest["files"].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        failures.append(name)
print(json.dumps({"verified_files": len(manifest["files"]), "changed_files": failures}, indent=2))
raise SystemExit(bool(failures))
