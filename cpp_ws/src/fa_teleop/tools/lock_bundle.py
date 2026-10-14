#!/usr/bin/python3
"""Create a relocatable source/model/config bundle and a compiled SHA-256 identity."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def generate(source, output, runtime):
    output.mkdir(parents=True, exist_ok=True)
    origins = json.loads((source / "ORIGINS.json").read_text())
    groups = {"ik": ["vendor/ik_7dof"], "min_snap": ["vendor/min_snap"],
              "fa_model": ["models"], "config": ["config"],
              "teleop": ["include", "src", "msg", "tools", "tests", "cmake", "launch", "CMakeLists.txt", "package.xml", "ORIGINS.json"]}
    files, components = {}, {}
    for group, roots in groups.items():
        names = []
        for root in roots:
            path = source / root
            names += ([path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()])
        entries = {}
        for path in sorted(names):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            name = path.relative_to(source).as_posix()
            entries[name] = sha(path.read_bytes())
            dest = output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        files.update(entries)
        source_name = {"ik": "ik_7dof", "min_snap": "min_snap", "fa_model": "fa_model"}.get(group)
        components[group] = {"version": "1.1.0" if group in {"teleop", "config"} else "1.0.0", "sha256": sha(json.dumps(entries, sort_keys=True).encode()),
                             "origin_commit": origins.get(source_name, {}).get("git_commit", "")}
    engine = json.loads(runtime.read_text())
    components["mujoco_runtime"] = {"version": engine["version"], "sha256": engine["library_sha256"], "origin_commit": ""}
    metadata = {"schema": 2, "version": "1.1.0", "components": components, "files": files, "mujoco_runtime": engine}
    raw = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode()
    release_id = sha(raw)
    (output / "release.json").write_bytes(raw)
    # Paths are relative to bundle root, so both installation and source builds are checked identically.
    (output / "files.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(files.items())))
    file_list_sha = sha((output / "files.sha256").read_bytes())
    (output / "bundle_version.hpp").write_text(
        '#pragma once\nnamespace fa {\n'
        f'inline constexpr const char* kReleaseId = "{release_id}";\n'
        f'inline constexpr const char* kFileListSha = "{file_list_sha}";\n'
        'inline constexpr const char* kVersion = "1.1.0";\n'
        f'inline constexpr const char* kMujocoLibrarySha = "{engine["library_sha256"]}";\n'
        f'inline constexpr const char* kMujocoVersion = "{engine["version"]}";\n}}\n')
    print("FA release:", release_id)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    args = p.parse_args()
    generate(args.source.resolve(), args.output.resolve(), args.runtime.resolve())
