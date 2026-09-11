"""Print the test-set results from artifacts/*/metrics.json as a Markdown table.

    python scripts/results_table.py            # capped truth (default, standard)
    python scripts/results_table.py --uncapped # against raw RUL_FD00x.txt values
"""

import argparse
import json

from aeropredict.config import ARTIFACT_DIR
from aeropredict.models import available_datasets

LABELS = {"linear": "Ridge (linear)", "lgbm": "LightGBM", "lstm": "LSTM"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--uncapped", action="store_true")
    args = p.parse_args()
    key = "test_uncapped_truth" if args.uncapped else "test"

    print("| Subset | Regimes | Sensors dropped | Model | RMSE | MAE | NASA score | Late % | Urgency acc. |")
    print("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name in available_datasets(ARTIFACT_DIR):
        m = json.loads((ARTIFACT_DIR / name / "metrics.json").read_text())
        best = min(m["models"], key=lambda k: m["models"][k][key]["nasa_score"])
        for i, (model, res) in enumerate(m["models"].items()):
            r = res[key]
            first = f"**{name}** | {m['n_regimes']} | {', '.join(m['sensors_dropped'])}" if i == 0 else " | | "
            label = f"**{LABELS[model]}**" if model == best else LABELS[model]
            print(
                f"| {first} | {label} | {r['rmse']:.2f} | {r['mae']:.2f} | {r['nasa_score']:.0f} "
                f"| {100 * r['late_fraction']:.0f}% | {100 * r['urgency_accuracy']:.0f}% |"
            )


if __name__ == "__main__":
    main()
