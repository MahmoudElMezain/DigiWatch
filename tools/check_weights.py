"""Check user-installed checkpoints against the filenames and optional SHA256 records supplied with this repository."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def canonical(name: str) -> str:
    """Input: model name; return: comparable model identifier."""
    return "".join(character for character in name.lower() if character.isalnum())


def check_weights(model: str | None = None, all_models: bool = False,
                  verify_hash: bool = False, directory: Path | None = None) -> bool:
    """Input: model and verification options; return: whether all requested files pass."""
    directory = directory or ROOT / "Models"
    manifest = json.loads((ROOT / "Models" / "weights_manifest.json").read_text(encoding="utf-8"))
    entries = manifest["weights"]
    segmentation = next(entry for entry in entries if entry["model"].startswith("CoralSCOP"))
    enhancements = {}
    for entry in entries:
        if entry is not segmentation:
            enhancements.setdefault(canonical(entry["model"]), []).append(entry)
    if all_models:
        requested = entries
    else:
        key = canonical(model or "mulagan")
        if key not in enhancements:
            raise ValueError(f"'{model}' is not available in the current pipeline. "
                             f"Available models: {', '.join(group[0]['model'] for group in enhancements.values())}.")
        requested = [segmentation, *enhancements[key]]
    success = True
    for entry in requested:
        path = directory / entry["path"]
        if not path.is_file() or path.stat().st_size == 0:
            print(f"MISSING  {entry['path']}")
            success = False
            continue
        if verify_hash:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                print(f"DIFFERS  {entry['path']} (does not match the tested checkpoint)")
                success = False
                continue
        status = "VERIFIED" if verify_hash else "PRESENT "
        print(f"{status} {entry['path']} ({path.stat().st_size / 1024**2:.1f} MiB)")
    if not success:
        print("See docs/MODELS.md and Models/weights_manifest.json for repository instructions and filenames.")
    elif not verify_hash:
        print("Files are present. Add --sha256 to compare them with the tested checkpoint hashes.")
    return success


def main() -> None:
    """Read checkpoint selection options and return a process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", help="Enhancement model; defaults to mulagan")
    selection.add_argument("--all", action="store_true", help="Check every implemented model's files")
    parser.add_argument("--sha256", action="store_true", help="Verify the tested checkpoint hashes")
    parser.add_argument("--directory", type=Path, help="Alternative Models directory")
    args = parser.parse_args()
    try:
        success = check_weights(args.model, args.all, args.sha256, args.directory)
    except ValueError as exc:
        parser.error(str(exc))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
