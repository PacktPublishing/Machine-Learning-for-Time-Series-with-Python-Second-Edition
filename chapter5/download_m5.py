"""Download the M5 Forecasting Accuracy dataset from Zenodo.

Data: M5 Forecasting Accuracy (Makridakis, Spiliotis & Assimakopoulos, 2022).
Licensed CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Zenodo record: https://zenodo.org/records/12636070

Usage:
    python download_m5.py            # download and extract into ./
    python download_m5.py --dest data  # download and extract into ./data
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

ZENODO_URL = (
    "https://zenodo.org/api/records/12636070/files/"
    "m5-forecasting-accuracy.zip/content"
)
EXPECTED_MD5 = "86f57416a314197f40a17cc6fc60cbb4"
EXPECTED_FILES = (
    "calendar.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
    "sample_submission.csv",
)


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream a URL to disk with a simple progress indicator."""
    with urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with dest.open("wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {downloaded / 1e6:6.1f} MB / "
                          f"{total / 1e6:6.1f} MB ({pct:5.1f}%)", end="")
    print()


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest", type=Path, default=Path("."),
        help="Directory to extract the data files into (default: current dir).",
    )
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)

    # Skip if all expected files are already present.
    if all((args.dest / name).exists() for name in EXPECTED_FILES):
        print("M5 files already present in", args.dest.resolve())
        return 0

    zip_path = args.dest / "m5-forecasting-accuracy.zip"
    if not zip_path.exists():
        print(f"Downloading M5 dataset from Zenodo ({EXPECTED_MD5[:8]}…) …")
        download(ZENODO_URL, zip_path)
    else:
        print(f"Found existing archive at {zip_path}, skipping download.")

    actual = md5sum(zip_path)
    if actual != EXPECTED_MD5:
        print(f"MD5 mismatch: got {actual}, expected {EXPECTED_MD5}",
              file=sys.stderr)
        return 1

    print(f"Extracting to {args.dest.resolve()} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.dest)

    print("Done. Files:")
    for name in EXPECTED_FILES:
        path = args.dest / name
        size_mb = path.stat().st_size / 1e6 if path.exists() else 0
        print(f"  {name:35s} {size_mb:6.1f} MB")

    print(
        "\nData licensed under CC-BY 4.0. Cite:\n"
        "  Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022).\n"
        "  M5 Forecasting Accuracy Dataset. Zenodo.\n"
        "  https://doi.org/10.5281/zenodo.12636070"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
