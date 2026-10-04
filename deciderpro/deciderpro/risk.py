"""Position sizing and safeguards. This module decides how long the bankroll survives."""
from dataclasses import dataclass


def kelly_fraction(p: float, odds: float) -> float:
    """Full-Kelly stake fraction for decimal odds. 0 when there is no edge."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (p * b - (1.0 - p)) / b)


@dataclass
class RiskConfig:
    kelly_mult: float = 0.25        # quarter-Kelly: gives up a little growth for far smaller drawdowns
    min_edge: float = 0.03          # required edge: p*odds - 1
    max_stake_frac: float = 0.02    # hard cap per bet as a share of current bankroll
    max_odds: float = 8.0           # skip long shots where model error dominates
    max_drawdown: float = 0.30      # halt when bankroll falls this far below its peak
    max_daily_frac: float = 0.06    # total new stakes per day, share of bankroll
    min_games: int = 8              # history required per team before betting
    max_model_prob_gap: float = 0.15  # distrust model if it disagrees with the market by more than this


class RiskManager:
    def __init__(self, cfg: RiskConfig, bankroll: float):
        self.cfg = cfg
        self.peak = bankroll
        self.halted = False
        self._day = None
        self._day_staked = 0.0

    def observe(self, bankroll: float) -> None:
        self.peak = max(self.peak, bankroll)
        if bankroll <= self.peak * (1.0 - self.cfg.max_drawdown):
            self.halted = True  # kill switch: stays off until a human resets it

    def stake(self, p: float, odds: float, market_p: float, bankroll: float, day: str) -> float:
        c = self.cfg
        if self.halted or odds > c.max_odds or odds <= 1.0:
            return 0.0
        if abs(p - market_p) > c.max_model_prob_gap:
            return 0.0
        if p * odds - 1.0 < c.min_edge:
            return 0.0
        if day != self._day:
            self._day, self._day_staked = day, 0.0
        frac = min(kelly_fraction(p, odds) * c.kelly_mult, c.max_stake_frac)
        room = c.max_daily_frac * bankroll - self._day_staked
        stake = max(0.0, min(frac * bankroll, room))
        self._day_staked += stake
        return stake
