"""Every market is priced and settled by the same `grade` rule over one result quantity.

Market keys (selection / line conventions):
  1x2              home | draw | away                      (football)
  moneyline        home | away                             (draws/ties settle as push)
  double_chance    1X | 12 | X2
  dnb              home | away   (draw no bet: push on a draw)
  spread           home | away, line from that side's view, e.g. home -1.5 / away +1.5
                   football = Asian handicap (half & whole lines), basketball/cricket = points/runs,
                   tennis = game handicap
  total            over | under, line                      (goals / points / runs / games)
  team_total_home, team_total_away   over | under, line
  btts             yes | no                                (both teams to score)
  correct_score    "2-1"  (football goals, tennis sets via set_betting)
  set_betting      "2-0", "2-1", ...                       (tennis)
  set_handicap     home | away, line                       (tennis, in sets)
  player_<stat>    "<Player> over" | "<Player> under", line   e.g. player_points, player_goals
                   (anytime scorer = player_goals over 0.5)
"""
from typing import Dict, Optional, Tuple

QUANTITY = {
    "1x2": "winner", "moneyline": "winner", "dnb": "winner", "double_chance": "winner",
    "spread": "margin", "total": "total", "team_total_home": "home", "team_total_away": "away",
    "btts": "btts", "correct_score": "score", "set_betting": "score", "set_handicap": "set_margin",
}


def grade(market: str, selection: str, line: Optional[float], value) -> str:
    """'win' | 'push' | 'lose' for one selection given the realised value of its quantity."""
    sel = selection.lower()
    if market == "1x2":
        return "win" if {"home": 1, "draw": 0, "away": -1}[sel] == value else "lose"
    if market in ("moneyline", "dnb"):
        if value == 0:
            return "push"
        return "win" if (value > 0) == (sel == "home") else "lose"
    if market == "double_chance":
        ok = {"1x": value >= 0, "12": value != 0, "x2": value <= 0}[sel]
        return "win" if ok else "lose"
    if market in ("spread", "set_handicap"):
        x = (value if sel == "home" else -value) + float(line)
        return "win" if x > 0 else "push" if x == 0 else "lose"
    if market == "btts":
        return "win" if bool(value) == (sel == "yes") else "lose"
    if market in ("correct_score", "set_betting"):
        return "win" if sel == str(value) else "lose"
    # over/under markets: totals, team totals, player props
    side = sel.rsplit(" ", 1)[-1]
    if side not in ("over", "under"):
        raise ValueError(f"cannot grade {market}/{selection}")
    diff = value - float(line)
    if diff == 0:
        return "push"
    return "win" if (diff > 0) == (side == "over") else "lose"


def player_market(market: str, selection: str) -> Optional[Tuple[str, str]]:
    """'player_points', 'L. James over' -> ('L. James', 'points')."""
    if not market.startswith("player_"):
        return None
    name = selection.rsplit(" ", 1)[0]
    return name, market[len("player_"):]


class Forecast:
    """Marginal distributions of every result quantity, plus player-prop distributions."""

    def __init__(self, marginals: Dict[str, Dict], meta: Dict = None, props=None):
        self.m = marginals
        self.meta = meta or {}
        self._props = props          # callable(player, stat) -> (dist, availability) | None

    def prob(self, market: str, selection: str, line: Optional[float] = None) -> Optional[Tuple[float, float, float]]:
        """(p_win, p_push, availability) or None if the market is not modelled for this sport."""
        avail = 1.0
        pm = player_market(market, selection)
        if pm:
            if not self._props:
                return None
            got = self._props(*pm)
            if not got:
                return None
            dist, avail = got
        else:
            q = QUANTITY.get(market)
            dist = self.m.get(q) if q else None
            if not dist:
                return None
        win = push = 0.0
        try:
            for v, p in dist.items():
                g = grade(market, selection, line, v)
                if g == "win":
                    win += p
                elif g == "push":
                    push += p
        except (KeyError, ValueError, TypeError):
            return None
        return win, push, avail

    def board(self) -> Dict[str, Dict[str, float]]:
        """Fair probabilities for the headline markets (for display)."""
        out: Dict[str, Dict[str, float]] = {}
        w = self.m.get("winner", {})
        if 0 in w and w[0] > 1e-3:
            out["1x2"] = {"home": w.get(1, 0), "draw": w.get(0, 0), "away": w.get(-1, 0)}
        else:
            out["moneyline"] = {"home": w.get(1, 0), "away": w.get(-1, 0)}
        for q in ("margin", "total"):
            d = self.m.get(q)
            if d:
                mean = sum(k * p for k, p in d.items())
                out[f"expected_{q}"] = {"mean": mean}
        if "btts" in self.m:
            out["btts"] = {"yes": self.m["btts"].get(True, 0.0)}
        if "score" in self.m:
            top = sorted(self.m["score"].items(), key=lambda kv: -kv[1])[:5]
            out["top_scores"] = dict(top)
        return out


def result_quantities(sport: str, result: Dict) -> Dict:
    """Realised values of every quantity, used for settlement."""
    h, a = result["home"], result["away"]
    q = {"winner": (h > a) - (h < a), "home": h, "away": a}
    if sport == "tennis":
        q["set_margin"] = h - a
        q["score"] = f"{h}-{a}"
        sets = result.get("sets")
        if sets:
            gh, ga = sum(s[0] for s in sets), sum(s[1] for s in sets)
            q.update(margin=gh - ga, total=gh + ga, home=gh, away=ga)
        else:
            q.pop("home"), q.pop("away")
    else:
        q.update(margin=h - a, total=h + a)
        if sport == "football":
            q["btts"] = h > 0 and a > 0
            q["score"] = f"{h}-{a}"
    return q


def settle(sport: str, market: str, selection: str, line: Optional[float], result: Dict) -> str:
    """'win' | 'push' | 'lose' | 'void' (void = cannot be settled / player did not play)."""
    pm = player_market(market, selection)
    if pm:
        stats = (result.get("player_stats") or {}).get(pm[0])
        if not stats or pm[1] not in stats:
            return "void"
        return grade(market, selection, line, stats[pm[1]])
    q = result_quantities(sport, result)
    key = QUANTITY.get(market)
    if key not in q:
        return "void"
    return grade(market, selection, line, q[key])
