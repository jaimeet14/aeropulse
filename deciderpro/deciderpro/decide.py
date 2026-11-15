"""Value finder: price every offered selection, compare with the de-margined market, rank by edge."""
from dataclasses import dataclass
from typing import Dict, List, Optional

from .data import Event, Odds
from .markets import Forecast, player_market
from .risk import RiskConfig


@dataclass
class Pick:
    odds: Odds
    p_win: float
    p_push: float
    market_p: float      # bookmaker's probability with margin removed
    edge: float          # expected profit per unit staked
    avail: float = 1.0
    stake: float = 0.0
    reason: str = ""


def _group_key(o: Odds):
    pm = player_market(o.market, o.selection)
    if pm:
        return (o.market, pm[0], o.line)
    if o.market in ("spread", "set_handicap"):
        return (o.market, o.line if o.selection.lower() == "home" else -o.line if o.line is not None else None)
    return (o.market, o.line)


def market_probs(odds: List[Odds]) -> Dict[int, float]:
    """Proportional de-margining within each market group. Single-sided offers assume a 5% margin."""
    groups: Dict[tuple, List[Odds]] = {}
    for o in odds:
        groups.setdefault(_group_key(o), []).append(o)
    out: Dict[int, float] = {}
    for key, os in groups.items():
        inv = [1.0 / o.price for o in os]
        expected = 2.0 if key[0] == "double_chance" else 1.0
        s = sum(inv) if len(os) > 1 else expected * 1.05
        for o, i in zip(os, inv):
            out[id(o)] = min(0.99, i * expected / s)
    return out


def evaluate(fc: Forecast, ev: Event, risk: RiskConfig) -> List[Pick]:
    """All offered selections with a modelled probability, best edge first. Includes rejected ones."""
    mp = market_probs(ev.odds)
    picks: List[Pick] = []
    for o in ev.odds:
        got = fc.prob(o.market, o.selection, o.line)
        if not got:
            continue
        pw, pp, avail = got
        edge = pw * o.price + pp - 1.0
        pk = Pick(o, pw, pp, mp[id(o)], edge, avail)
        cond = pw / (1.0 - pp) if pp < 1 else 0.0
        if avail < risk.min_player_avail:
            pk.reason = f"player availability {avail:.0%}"
        elif abs(cond - pk.market_p) > risk.max_model_prob_gap:
            pk.reason = "model disagrees with market too much"
        elif o.price > risk.max_odds:
            pk.reason = "odds above cap"
        elif edge < risk.min_edge:
            pk.reason = "edge below threshold"
        picks.append(pk)
    picks.sort(key=lambda p: -p.edge)
    return picks
