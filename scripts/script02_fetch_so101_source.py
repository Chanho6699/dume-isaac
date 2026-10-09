#!/usr/bin/env python3
"""Stage 1a: fetch / verify the official SO-101 source files pinned in source_manifest.yaml.

Downloads only the files listed in dume_isaac/assets/my_so101/source_manifest.yaml from
the pinned TheRobotStudio/SO-ARM100 commit (raw.githubusercontent.com) and checks each
sha256. Existing files with the right hash are not downloaded again.

Python stdlib + PyYAML only (works with or without Isaac).

    python scripts/script02_fetch_so101_source.py              # fetch missing + verify
    python scripts/script02_fetch_so101_source.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dume_isaac.config_io import load_yaml  # noqa: E402
from dume_isaac.paths import MY_SO101_DIR, repo_path  # noqa: E402

MANIFEST = MY_SO101_DIR / "source_manifest.yaml"
OFFICIAL_REPO = "https://github.com/TheRobotStudio/SO-ARM100"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(m: dict) -> list[str]:
    """Return a list of problems (empty = valid). Pure; used by tests too."""
    errors = []
    up = m.get("upstream") or {}
    if up.get("repository") != OFFICIAL_REPO:
        errors.append(f"upstream.repository must be {OFFICIAL_REPO} (official source only)")
    if not _COMMIT_RE.match(str(up.get("commit", ""))):
        errors.append("upstream.commit must be a full 40-char commit hash")
    files = m.get("files") or []
    if not files:
        errors.append("files list is empty")
    for i, f in enumerate(files):
        for key in ("upstream_path", "local_path", "sha256"):
            if not f.get(key):
                errors.append(f"files[{i}].{key} missing")
        if f.get("sha256") and not re.fullmatch(r"[0-9a-f]{64}", str(f["sha256"])):
            errors.append(f"files[{i}].sha256 is not a sha256 hex digest")
        if f.get("local_path") and not str(f["local_path"]).startswith("dume_isaac/assets/my_so101/"):
            errors.append(f"files[{i}].local_path must stay under dume_isaac/assets/my_so101/")
    sel = (m.get("selected_model") or {}).get("file")
    if sel and not any(str(f.get("upstream_path", "")).endswith("/" + sel) for f in files):
        errors.append(f"selected_model.file {sel} is not in files")
    return errors


def raw_url(commit: str, upstream_path: str) -> str:
    return f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{commit}/{upstream_path}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch/verify pinned official SO-101 source files.")
    parser.add_argument("--verify-only", action="store_true", help="do not download, only check hashes")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    m = load_yaml(args.manifest)
    problems = validate_manifest(m)
    if problems:
        print("Manifest invalid:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    commit = m["upstream"]["commit"]
    print(f"=== SO-101 source: {m['upstream']['repository']} @ {commit[:12]} ({m['upstream'].get('license')}) ===")

    ok = bad = fetched = 0
    for f in m["files"]:
        dest = repo_path(f["local_path"])
        if dest.is_file() and sha256(dest) == f["sha256"]:
            ok += 1
            print(f"  OK       {f['local_path']}")
            continue
        if args.verify_only:
            bad += 1
            print(f"  MISSING  {f['local_path']}" if not dest.is_file() else f"  MISMATCH {f['local_path']}")
            continue
        url = raw_url(commit, f["upstream_path"])
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except Exception as exc:
            bad += 1
            print(f"  FAIL     {f['upstream_path']}: {type(exc).__name__}: {exc}")
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest != f["sha256"]:
            bad += 1
            print(f"  FAIL     {f['upstream_path']}: sha256 {digest[:12]} != manifest {f['sha256'][:12]} (not written)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        fetched += 1
        print(f"  FETCHED  {f['local_path']}")

    total = len(m["files"])
    print(f"\nSummary: {ok} verified, {fetched} fetched, {bad} problems, {total} total")
    if bad:
        print("FAIL: fix network access or the manifest, then re-run.", file=sys.stderr)
        return 1
    print("PASS: all source files match the pinned upstream revision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
