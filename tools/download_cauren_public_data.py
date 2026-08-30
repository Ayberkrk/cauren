from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MANIFEST = Path("data/public_sources/source_manifest.json")
DEFAULT_OUTPUT_DIR = Path("data/public_sources/raw")
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; CaurenCivilResearchBot/1.0; +https://example.invalid/cauren-civil)"


def download_public_sources(*, manifest_path: Path, output_dir: Path, only: set[str] | None = None) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    downloads: list[dict[str, object]] = []
    for source in manifest.get("sources", []):
        source_id = str(source.get("source_id") or "").strip()
        if not source_id:
            continue
        if only and source_id not in only:
            continue
        download_url = str(source.get("download_url") or "").strip()
        if not download_url:
            downloads.append({"source_id": source_id, "status": "skipped", "reason": "no_download_url"})
            continue
        target_name = str(source.get("target_name") or f"{source_id}.bin").strip()
        target_path = output_dir / target_name
        digest = hashlib.sha256()
        request = urllib.request.Request(download_url, headers={"User-Agent": DEFAULT_USER_AGENT})
        try:
            with urllib.request.urlopen(request) as response, target_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 64)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
        except urllib.error.HTTPError as exc:
            downloads.append(
                {
                    "source_id": source_id,
                    "status": "failed",
                    "reason": f"http_{exc.code}",
                    "download_url": download_url,
                }
            )
            continue
        except urllib.error.URLError as exc:
            downloads.append(
                {
                    "source_id": source_id,
                    "status": "failed",
                    "reason": f"url_error:{exc.reason}",
                    "download_url": download_url,
                }
            )
            continue
        downloads.append(
            {
                "source_id": source_id,
                "status": "downloaded",
                "target_path": str(target_path),
                "sha256": digest.hexdigest(),
                "license_note": source.get("license_note", ""),
            }
        )
    summary = {"manifest": str(manifest_path), "downloads": downloads}
    (output_dir / "download_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def copy_manifest_examples(*, manifest_path: Path, output_dir: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in manifest.get("sources", []):
        example_path = str(source.get("bundled_example") or "").strip()
        if not example_path:
            continue
        src = manifest_path.parent / example_path
        if src.exists():
            shutil.copy2(src, output_dir / src.name)
            copied += 1
    summary = {"copied_examples": copied}
    (output_dir / "example_copy_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public civil engineering source files listed in the manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--copy-bundled-examples", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    if args.copy_bundled_examples:
        summary = copy_manifest_examples(manifest_path=manifest_path, output_dir=output_dir)
    else:
        selected = {item for item in args.only if item}
        summary = download_public_sources(manifest_path=manifest_path, output_dir=output_dir, only=selected or None)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
