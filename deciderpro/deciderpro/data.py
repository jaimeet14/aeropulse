"""Match records and CSV loading."""
import csv
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


@dataclass
class Match:
    date: str                      # ISO date, sorted lexicographically
    home: str                      # home team / player A
    away: str                      # away team / player B
    result: int                    # index into odds: 3-way 0=home 1=draw 2=away; 2-way 0=home 1=away
    odds: Sequence[float]          # decimal odds available at bet time, aligned to outcomes
    closing_odds: Optional[Sequence[float]] = None  # for closing-line-value tracking


def _f(row, *keys) -> Optional[float]:
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                pass
    return None


def load_football_data_csv(path: str) -> List[Match]:
    """football-data.co.uk format: Date,HomeTeam,AwayTeam,FTR,B365H/D/A (+ B365CH/CD/CA closing)."""
    out: List[Match] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            o = [_f(row, "B365H", "AvgH"), _f(row, "B365D", "AvgD"), _f(row, "B365A", "AvgA")]
            c = [_f(row, "B365CH", "AvgCH"), _f(row, "B365CD", "AvgCD"), _f(row, "B365CA", "AvgCA")]
            ftr = (row.get("FTR") or "").strip().upper()
            if None in o or ftr not in ("H", "D", "A"):
                continue
            d = row["Date"].strip()
            if "/" in d:  # dd/mm/yy[yy] -> ISO
                dd, mm, yy = d.split("/")
                yy = ("20" + yy) if len(yy) == 2 else yy
                d = f"{yy}-{mm.zfill(2)}-{dd.zfill(2)}"
            out.append(Match(d, row["HomeTeam"], row["AwayTeam"], "HDA".index(ftr), o,
                             c if None not in c else None))
    out.sort(key=lambda m: m.date)
    return out


def load_generic_csv(path: str) -> List[Match]:
    """Generic 2-way format: date,home,away,winner(home|away),odds_home,odds_away[,close_home,close_away]."""
    out: List[Match] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            w = row["winner"].strip().lower()
            if w not in ("home", "away"):
                continue
            o = [float(row["odds_home"]), float(row["odds_away"])]
            ch, ca = _f(row, "close_home"), _f(row, "close_away")
            out.append(Match(row["date"], row["home"], row["away"], 0 if w == "home" else 1, o,
                             [ch, ca] if ch and ca else None))
    out.sort(key=lambda m: m.date)
    return out
