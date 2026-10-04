"""Walk-forward backtest: predict -> decide -> settle -> learn, strictly in date order.

The model only ever sees matches that were already played, so there is no look-ahead.
"""
import math
from dataclasses import dataclass, field
from typing import List, Sequence

from .data import Match
from .model import RatingModel
from .risk import RiskConfig, RiskManager
from .sports import SportConfig


def implied_probs(odds: Sequence[float]) -> List[float]:
    """Market probabilities with the bookmaker margin removed (proportional normalisation)."""
    inv = [1.0 / o for o in odds]
    s = sum(inv)
    return [x / s for x in inv]


@dataclass
class Report:
    start_bankroll: float
    end_bankroll: float = 0.0
    bets: int = 0
    wins: int = 0
    staked: float = 0.0
    max_drawdown: float = 0.0
    halted: bool = False
    matches: int = 0
    model_logloss: float = 0.0
    market_logloss: float = 0.0
    clv_sum: float = 0.0
    clv_n: int = 0
    curve: List[float] = field(default_factory=list)

    @property
    def roi(self) -> float:
        return (self.end_bankroll - self.start_bankroll) / self.staked if self.staked else 0.0

    @property
    def avg_clv(self) -> float:
        return self.clv_sum / self.clv_n if self.clv_n else float("nan")

    def summary(self) -> str:
        growth = self.end_bankroll / self.start_bankroll - 1.0
        lines = [
            f"matches seen        : {self.matches}",
            f"bets placed         : {self.bets}  (won {self.wins})",
            f"bankroll            : {self.start_bankroll:.2f} -> {self.end_bankroll:.2f}  ({growth:+.1%})",
            f"yield (profit/staked): {self.roi:+.2%}",
            f"max drawdown        : {self.max_drawdown:.1%}",
            f"kill switch tripped : {self.halted}",
            f"model log-loss      : {self.model_logloss:.4f}",
            f"market log-loss     : {self.market_logloss:.4f}   (model must beat this to have an edge)",
            f"avg closing-line value: {self.avg_clv:+.2%}  over {self.clv_n} bets" if self.clv_n
            else "avg closing-line value: n/a (no closing odds in data)",
        ]
        return "\n".join(lines)


def run_backtest(matches: List[Match], cfg: SportConfig, bankroll: float = 1000.0,
                 risk: RiskConfig = None, lr: float = 0.01) -> Report:
    risk = risk or RiskConfig()
    model = RatingModel(cfg, lr=lr)
    rm = RiskManager(risk, bankroll)
    rep = Report(start_bankroll=bankroll)
    peak = bankroll
    ll_m = ll_k = 0.0
    scored = 0

    for m in matches:
        probs = model.predict(m.home, m.away)
        mkt = implied_probs(m.odds)

        if model.confidence(m.home, m.away, risk.min_games):
            scored += 1
            ll_m += -math.log(max(probs[m.result], 1e-9))
            ll_k += -math.log(max(mkt[m.result], 1e-9))
            # pick the single outcome with the best edge
            best = max(range(len(probs)), key=lambda i: probs[i] * m.odds[i])
            stake = rm.stake(probs[best], m.odds[best], mkt[best], bankroll, m.date)
            if stake > 0:
                rep.bets += 1
                rep.staked += stake
                if m.result == best:
                    rep.wins += 1
                    bankroll += stake * (m.odds[best] - 1.0)
                else:
                    bankroll -= stake
                if m.closing_odds:  # beat the closing price => positive CLV
                    rep.clv_sum += m.odds[best] / m.closing_odds[best] - 1.0
                    rep.clv_n += 1

        model.update(m.home, m.away, m.result)  # learn only after settling
        rm.observe(bankroll)
        peak = max(peak, bankroll)
        rep.max_drawdown = max(rep.max_drawdown, 1.0 - bankroll / peak)
        rep.curve.append(bankroll)
        rep.matches += 1

    rep.end_bankroll = bankroll
    rep.halted = rm.halted
    if scored:
        rep.model_logloss, rep.market_logloss = ll_m / scored, ll_k / scored
    return rep
