#!/usr/bin/python3
"""Human-readable component versions derived from the installed, compiled manifest."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--prefix", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
bundle = args.prefix / "share/fa_teleop/bundle"
manifest = json.loads((bundle / "release.json").read_text())
report = {"release_id": hashlib.sha256((bundle / "release.json").read_bytes()).hexdigest(),
          "components": {}, "profile_sha256": {}}
for name, component in manifest["components"].items():
    report["components"][name] = {**component, "full_version": component["version"] + "+" + component["sha256"][:12]}
for profile in sorted((bundle / "config").glob("*.yaml")):
    report["profile_sha256"][profile.name] = hashlib.sha256(profile.read_bytes()).hexdigest()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print("Component versions:", args.output)
