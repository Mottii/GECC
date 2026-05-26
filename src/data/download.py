import argparse
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from src.utils.config import DATA_URL, RAW_DIR, ensure_project_dirs


def find_raw_csvs(output_dir: Path = RAW_DIR) -> tuple[Path | None, Path | None]:
    data_candidates = sorted(output_dir.rglob("data.csv"))
    labels_candidates = sorted(output_dir.rglob("labels.csv"))
    data_path = data_candidates[0] if data_candidates else None
    labels_path = labels_candidates[0] if labels_candidates else None
    return data_path, labels_path


def download_file(url: str = DATA_URL, output_dir: Path = RAW_DIR) -> Path:
    ensure_project_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / Path(url).name

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with target.open("wb") as handle:
        with tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))
    return target


def extract_archive(archive_path: Path, output_dir: Path = RAW_DIR) -> None:
    if not tarfile.is_tarfile(archive_path):
        raise ValueError(f"Unsupported archive format: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(output_dir)


def ensure_raw_data(url: str = DATA_URL, output_dir: Path = RAW_DIR, force: bool = False) -> None:
    data_csv, labels_csv = find_raw_csvs(output_dir)
    if data_csv and labels_csv and not force:
        print(f"Raw dataset already exists: {data_csv} and {labels_csv}")
        return

    archive_path = download_file(url=url, output_dir=output_dir)
    extract_archive(archive_path, output_dir=output_dir)
    data_csv, labels_csv = find_raw_csvs(output_dir)
    if not data_csv or not labels_csv:
        raise FileNotFoundError(
            f"Expected data.csv and labels.csv after extraction in {output_dir}"
        )
    print(f"Dataset ready: {data_csv} and {labels_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the UCI RNA-Seq dataset.")
    parser.add_argument("--url", default=DATA_URL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_raw_data(url=args.url, force=args.force)


if __name__ == "__main__":
    main()
