"""Download NASA C-MAPSS into data/raw (idempotent)."""

import argparse
import logging

from aeropredict.config import DATA_DIR
from aeropredict.download import download_cmapss

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = p.parse_args()
    download_cmapss(DATA_DIR, force=args.force)
    print(f"Data ready in {DATA_DIR}")
