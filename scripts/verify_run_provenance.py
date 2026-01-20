from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_run(run_dir: Path) -> int:
    meta_path = run_dir / "run_metadata.json"
    stamp_path = run_dir / "run_metadata.sha256"

    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))

    meta_bytes = meta_path.read_bytes()
    meta = json.loads(meta_bytes.decode("utf-8"))

    errors: list[str] = []

    # Verify stamp (if present)
    if stamp_path.exists():
        expected = stamp_path.read_text(encoding="utf-8").strip()
        actual = hashlib.sha256(meta_bytes).hexdigest()
        if expected and expected != actual:
            errors.append("run_metadata.sha256 does not match run_metadata.json (metadata edited)")

    # Verify inputs
    inputs = meta.get("inputs") or {}
    for name, info in inputs.items():
        p = Path(info.get("path", ""))
        if not p.exists():
            errors.append(f"Missing input file: {name} -> {p}")
            continue
        expected_hash = info.get("sha256")
        if expected_hash:
            actual_hash = sha256_file(p)
            if actual_hash != expected_hash:
                errors.append(f"Input hash mismatch: {name} -> {p}")

    # Verify outputs manifest
    out_manifest = meta.get("outputs_manifest") or {}
    files = out_manifest.get("files") or []
    for info in files:
        p = Path(info.get("path", ""))
        if not p.exists():
            errors.append(f"Missing output file: {p}")
            continue
        expected_hash = info.get("sha256")
        if expected_hash:
            actual_hash = sha256_file(p)
            if actual_hash != expected_hash:
                errors.append(f"Output hash mismatch: {p}")

    if errors:
        print("PROVENANCE CHECK: FAIL")
        for e in errors:
            print("-", e)
        return 1

    print("PROVENANCE CHECK: PASS")
    git = meta.get("git")
    if git and git.get("head"):
        print("git head:", git.get("head"))
        print("git dirty:", git.get("dirty"))

    print("verified inputs:", len(inputs))
    print("verified outputs:", len(files))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to a run output directory (e.g. outputs/sample_run)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    raise SystemExit(verify_run(run_dir))


if __name__ == "__main__":
    main()
