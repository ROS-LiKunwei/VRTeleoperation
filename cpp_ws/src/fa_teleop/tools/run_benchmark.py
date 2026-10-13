#!/usr/bin/env python3
"""One-factor parameter sweeps; preserve every result and choose by declared criteria."""
import argparse
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--binary", required=True)
p.add_argument("--bundle", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--steps", type=int, default=4000)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
records = []
for duration in [0.02, 0.04, 0.06, 0.08]:
    path = args.output / f"duration_{duration:.2f}.json"
    result = subprocess.run([args.binary, "--bundle", args.bundle, "--steps", str(args.steps),
                             "--duration", str(duration), "--output", str(path)], capture_output=True, text=True)
    if not path.exists():
        raise RuntimeError(result.stdout + result.stderr)
    record = json.loads(path.read_text())
    record["returncode"] = result.returncode
    records.append(record)
eligible = [r for r in records if r["accuracy_pass"] and r["failed_steps"] == 0 and r["compute_ms_p99"] < 5]
best = min(eligible, key=lambda r: max(r["position_max_m"])) if eligible else None
(args.output / "selection.json").write_text(json.dumps({"criteria": "minimize maximum simultaneous TCP position error among zero-failure precision-qualified candidates with compute p99 below 5 ms; hardware unverified", "best": best, "candidates": records}, indent=2) + "\n")
print("Selected duration:", best["trajectory_duration_s"] if best else "none")
raise SystemExit(0 if best else 2)
