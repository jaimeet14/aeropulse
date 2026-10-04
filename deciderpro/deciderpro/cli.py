import argparse

from . import data, synthetic
from .backtest import run_backtest
from .risk import RiskConfig
from .sports import SPORTS, get_sport


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="deciderpro", description="Sports value-bet backtester (simulation only).")
    ap.add_argument("--sport", choices=sorted(SPORTS), default="football")
    ap.add_argument("--csv", help="historical data file (football-data.co.uk format for football, generic format otherwise)")
    ap.add_argument("--synthetic", action="store_true", help="use generated data to test the pipeline")
    ap.add_argument("--matches", type=int, default=3000, help="synthetic match count")
    ap.add_argument("--book-noise", type=float, default=60.0, help="synthetic bookmaker error in Elo points (0 = perfect book)")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--kelly", type=float, default=0.25, help="Kelly multiplier, 0.25 = quarter Kelly")
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--max-drawdown", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    cfg = get_sport(a.sport)
    if a.synthetic:
        matches = synthetic.generate(cfg, a.matches, book_noise=a.book_noise, seed=a.seed)
        print("** SYNTHETIC DATA: checks the machinery only, says nothing about real profitability **\n")
    elif a.csv:
        loader = data.load_football_data_csv if a.sport == "football" else data.load_generic_csv
        matches = loader(a.csv)
    else:
        ap.error("provide --csv FILE or --synthetic")

    risk = RiskConfig(kelly_mult=a.kelly, min_edge=a.min_edge, max_drawdown=a.max_drawdown)
    print(run_backtest(matches, cfg, a.bankroll, risk).summary())
    return 0
