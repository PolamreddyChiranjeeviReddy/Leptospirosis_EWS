from __future__ import annotations

import json
import platform
import sys
import hashlib
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_info(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": _sha256_file(path),
    }


def _safe_git_info(repo_dir: Path) -> dict | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0 or res.stdout.strip().lower() != "true":
            return None

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        ).stdout

        return {
            "head": head or None,
            "dirty": bool(status.strip()),
            "status_porcelain": status.splitlines(),
        }
    except Exception:
        return None


def _iter_files_recursively(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            files.append(p)
    return files


def write_run_metadata(
    output_dir: Path,
    *,
    config_dict: dict,
    input_files: dict[str, str | None] | None = None,
    repo_dir: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if repo_dir is None:
        # best effort: assume repo root is two levels above this file
        repo_dir = Path(__file__).resolve().parents[2]

    meta: dict = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": {
            # keep this list short + relevant
            "numpy": _pkg_version("numpy"),
            "pandas": _pkg_version("pandas"),
            "scikit-learn": _pkg_version("scikit-learn"),
            "xgboost": _pkg_version("xgboost"),
            "shap": _pkg_version("shap"),
            "geopandas": _pkg_version("geopandas"),
            "folium": _pkg_version("folium"),
            "pydantic": _pkg_version("pydantic"),
            "typer": _pkg_version("typer"),
        },
        "git": _safe_git_info(repo_dir) if repo_dir is not None else None,
        "inputs": {},
        "config": config_dict,
    }

    if input_files:
        inputs: dict[str, dict] = {}
        for key, p in input_files.items():
            if not p:
                continue
            path = Path(p)
            if not path.exists():
                inputs[key] = {"path": str(path), "exists": False}
                continue
            inputs[key] = _file_info(path)
        meta["inputs"] = inputs

    path = output_dir / "run_metadata.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return path


def finalize_run_metadata(output_dir: Path) -> Path:
    """Add a hashed manifest of all generated artifacts under output_dir.

    This makes post-run manual edits detectable (hash mismatch).
    """
    meta_path = output_dir / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    files = _iter_files_recursively(output_dir)
    manifest: list[dict] = []
    for p in sorted(files, key=lambda x: str(x).lower()):
        # avoid self-referential hashing
        if p.resolve() == meta_path.resolve():
            continue
        if p.name.lower() == "run_metadata.sha256":
            continue
        manifest.append(_file_info(p))

    meta["outputs_manifest"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": manifest,
    }

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    # Write a simple stamp: sha256 of the metadata file as-written.
    stamp = output_dir / "run_metadata.sha256"
    stamp.write_text(_sha256_file(meta_path) + os.linesep, encoding="utf-8")
    return meta_path
