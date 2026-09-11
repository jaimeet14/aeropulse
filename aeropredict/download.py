"""Fetch the NASA C-MAPSS dataset. No API key required.

The public mirror ships a zip that contains another zip (``CMAPSSData.zip``) with the
actual ``train_FD00x.txt`` / ``test_FD00x.txt`` / ``RUL_FD00x.txt`` files, so we
unpack recursively until the text files land in ``DATA_DIR``.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

from .config import DATA_DIR, DATASETS, DOWNLOAD_URL

log = logging.getLogger(__name__)

EXPECTED_FILES = [
    f"{kind}_{ds}.txt" for ds in DATASETS for kind in ("train", "test", "RUL")
]


def dataset_present(data_dir: Path = DATA_DIR, datasets: tuple[str, ...] = DATASETS) -> bool:
    files = [f"{kind}_{ds}.txt" for ds in datasets for kind in ("train", "test", "RUL")]
    return all((data_dir / f).exists() for f in files)


def _extract_recursive(blob: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            data = zf.read(member)
            if name.lower().endswith(".zip"):
                _extract_recursive(data, dest)
            elif name.endswith(".txt"):
                (dest / name).write_bytes(data)


def download_cmapss(data_dir: Path = DATA_DIR, url: str = DOWNLOAD_URL, force: bool = False) -> Path:
    """Download and unpack C-MAPSS into ``data_dir``. Idempotent unless ``force``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if dataset_present(data_dir) and not force:
        log.info("C-MAPSS already present in %s", data_dir)
        return data_dir

    cached = data_dir / "cmapss.zip"
    if cached.exists() and not force:
        blob = cached.read_bytes()
    else:
        log.info("Downloading C-MAPSS from %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "aeropredict/0.1"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            blob = resp.read()
        cached.write_bytes(blob)

    _extract_recursive(blob, data_dir)
    missing = [f for f in EXPECTED_FILES if not (data_dir / f).exists()]
    if missing:
        raise RuntimeError(f"Download finished but files are missing: {missing}")
    log.info("C-MAPSS ready in %s", data_dir)
    return data_dir
