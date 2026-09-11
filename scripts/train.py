"""Train and evaluate AeroPredict models on one or more C-MAPSS sub-datasets.

    python scripts/train.py --dataset FD001
    python scripts/train.py --dataset all --epochs 40
"""

import argparse
import json
import logging

from aeropredict.config import ARTIFACT_DIR, DATA_DIR, DATASETS
from aeropredict.data import load_subset
from aeropredict.download import download_cmapss
from aeropredict.models import save_bundle
from aeropredict.train import train_subset


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="FD001", help="FD001..FD004 or 'all'")
    p.add_argument("--models", default="linear,lgbm,lstm")
    p.add_argument("--epochs", type=int, default=40, help="max LSTM epochs")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    args = p.parse_args()

    download_cmapss(DATA_DIR)
    names = DATASETS if args.dataset.lower() == "all" else [args.dataset.upper()]
    models = tuple(m.strip() for m in args.models.split(","))

    for name in names:
        subset = load_subset(name, DATA_DIR)
        bundle = train_subset(subset, models=models, lstm_epochs=args.epochs, device=args.device)
        out = save_bundle(bundle, ARTIFACT_DIR)
        print(f"\n=== {name} -> {out}")
        for m, res in bundle.metrics["models"].items():
            print(f"  {m:7s} {json.dumps(res['test'])}")


if __name__ == "__main__":
    main()
