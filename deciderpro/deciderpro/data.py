"""Event / odds records, JSON-lines storage and CSV importers.

Event JSON (one per line in a .jsonl file):
{
  "sport": "basketball", "date": "2026-01-14", "home": "BOS", "away": "NYK",
  "context":  {"rest_home": 1, "rest_away": 2, "neutral": false, "venue": "TD Garden",
               "surface": "hard", "best_of": 3, "format": "t20", "toss": "home"},
  "injuries": [{"player": "J. Tatum", "team": "BOS", "status": "questionable"}],
  "lineups":  {"home": ["..."], "away": ["..."]},           # optional, when announced
  "odds":     [{"market": "spread", "selection": "home", "line": -4.5, "price": 1.91,
                "closing": 1.87}],
  "result":   {"home": 112, "away": 104, "sets": [[6, 4], [3, 6], [7, 6]],
               "player_stats": {"J. Brown": {"points": 27, "rebounds": 6}}}
}
"""
import csv
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class Odds:
    market: str
    selection: str
    price: float
    line: Optional[float] = None
    closing: Optional[float] = None
    book: str = ""


@dataclass
class Event:
    sport: str
    date: str
    home: str
    away: str
    context: Dict = field(default_factory=dict)
    injuries: List[Dict] = field(default_factory=list)
    lineups: Optional[Dict[str, List[str]]] = None
    odds: List[Odds] = field(default_factory=list)
    result: Optional[Dict] = None
    id: str = ""

    @classmethod
    def from_dict(cls, d: Dict) -> "Event":
        d = dict(d)
        d["odds"] = [o if isinstance(o, Odds) else Odds(**o) for o in d.get("odds", [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)


def load_jsonl(path: str) -> List[Event]:
    with open(path, encoding="utf-8") as fh:
        evs = [Event.from_dict(json.loads(line)) for line in fh if line.strip()]
    evs.sort(key=lambda e: e.date)
    return evs


def save_jsonl(events: List[Event], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e.to_dict()) + "\n")


def _f(row, *keys) -> Optional[float]:
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                pass
    return None


def load_football_data_csv(path: str) -> List[Event]:
    """football-data.co.uk CSV -> events with 1X2, over/under 2.5 and Asian handicap odds.

    Uses Bet365 prices where present, else market averages; closing prices (B365C*/AvgC*) feed CLV.
    """
    out: List[Event] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                hg, ag = int(row["FTHG"]), int(row["FTAG"])
            except (KeyError, ValueError):
                continue
            d = row["Date"].strip()
            if "/" in d:
                dd, mm, yy = d.split("/")
                yy = ("20" + yy) if len(yy) == 2 else yy
                d = f"{yy}-{mm.zfill(2)}-{dd.zfill(2)}"
            odds: List[Odds] = []

            def add(market, sel, line, open_keys, close_keys):
                p = _f(row, *open_keys)
                if p and p > 1.0:
                    odds.append(Odds(market, sel, p, line, _f(row, *close_keys)))

            add("1x2", "home", None, ("B365H", "AvgH"), ("B365CH", "AvgCH"))
            add("1x2", "draw", None, ("B365D", "AvgD"), ("B365CD", "AvgCD"))
            add("1x2", "away", None, ("B365A", "AvgA"), ("B365CA", "AvgCA"))
            add("total", "over", 2.5, ("B365>2.5", "Avg>2.5"), ("B365C>2.5", "AvgC>2.5"))
            add("total", "under", 2.5, ("B365<2.5", "Avg<2.5"), ("B365C<2.5", "AvgC<2.5"))
            ah = _f(row, "AHh")
            if ah is not None:
                ahc = _f(row, "AHCh")
                # closing AH prices only comparable when the closing line is unchanged
                same = ahc is None or ahc == ah
                add("spread", "home", ah, ("B365AHH", "AvgAHH"), ("B365CAHH", "AvgCAHH") if same else ())
                add("spread", "away", -ah, ("B365AHA", "AvgAHA"), ("B365CAHA", "AvgCAHA") if same else ())
            out.append(Event("football", d, row["HomeTeam"], row["AwayTeam"], odds=odds,
                             result={"home": hg, "away": ag}))
    out.sort(key=lambda e: e.date)
    return out
