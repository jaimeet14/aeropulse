"""Position sizing and safeguards. This module decides how long the bankroll survives."""
from dataclasses import dataclass
from typing import Dict


def kelly_fraction(p: float, odds: float, push: float = 0.0) -> float:
    """Full-Kelly stake fraction for decimal odds, with an optional push (stake returned) probability."""
    b = odds - 1.0
    q = 1.0 - p - push
    if b <= 0 or p <= 0:
        return 0.0
    return max(0.0, (p * b - q) / (b * (p + q)))


@dataclass
class RiskConfig:
    kelly_mult: float = 0.25          # quarter-Kelly: gives up a little growth for far smaller drawdowns
    min_edge: float = 0.03            # required expected value per unit staked
    max_stake_frac: float = 0.02      # hard cap per bet, share of bankroll
    max_event_frac: float = 0.03      # all bets on one event combined (markets on one game are correlated)
    max_bets_per_event: int = 2
    max_daily_frac: float = 0.10      # new stakes per day
    max_odds: float = 8.0             # long shots: model error dominates
    max_drawdown: float = 0.30        # kill switch: halt when this far below peak
    min_games: int = 8                # history required per team/player before betting
    max_model_prob_gap: float = 0.12  # distrust the model when it disagrees with the market this much
    min_player_avail: float = 0.95    # props only when the player is confirmed to play


class RiskManager:
    def __init__(self, cfg: RiskConfig, bankroll: float):
        self.cfg = cfg
        self.peak = bankroll
        self.halted = False
        self._day = None
        self._day_staked = 0.0
        self._event: Dict[str, list] = {}

    def observe(self, bankroll: float) -> None:
        self.peak = max(self.peak, bankroll)
        if bankroll <= self.peak * (1.0 - self.cfg.max_drawdown):
            self.halted = True  # stays off until a human resets it

    def stake(self, p: float, push: float, odds: float, bankroll: float, day: str, event: str) -> float:
        c = self.cfg
        if self.halted or odds > c.max_odds or odds <= 1.0:
            return 0.0
        if p * odds + push - 1.0 < c.min_edge:
            return 0.0
        if day != self._day:
            self._day, self._day_staked, self._event = day, 0.0, {}
        n, used = self._event.get(event, (0, 0.0))
        if n >= c.max_bets_per_event:
            return 0.0
        frac = min(kelly_fraction(p, odds, push) * c.kelly_mult, c.max_stake_frac)
        stake = min(frac * bankroll,
                    c.max_daily_frac * bankroll - self._day_staked,
                    c.max_event_frac * bankroll - used)
        stake = max(0.0, stake)
        if stake > 0:
            self._day_staked += stake
            self._event[event] = (n + 1, used + stake)
        return stake


class TrustGate:
    """Self-assessment: bet a market type only while the model out-predicts the bookmaker there.

    For every offered selection that settles, it records Brier(model) - Brier(market) per market
    type over a rolling window. Betting in that market is allowed only after `min_obs` observations
    and while the model is better by at least `z` standard errors (mean + z*se < 0), so a lucky
    streak in a market where the model is merely as good as the book does not open it.
    This is what makes the system stop on its own in markets where it has no edge.
    """

    def __init__(self, window: int = 400, min_obs: int = 200, z: float = 1.0):
        self.window, self.min_obs, self.z = window, min_obs, z
        self.hist: Dict[str, list] = {}

    def record(self, market: str, p_model: float, p_market: float, won: bool) -> None:
        y = 1.0 if won else 0.0
        h = self.hist.setdefault(market, [])
        h.append((p_model - y) ** 2 - (p_market - y) ** 2)
        if len(h) > self.window:
            del h[: len(h) - self.window]

    def score(self, market: str):
        h = self.hist.get(market, [])
        return (sum(h) / len(h) if h else 0.0), len(h)

    def allows(self, market: str) -> bool:
        h = self.hist.get(market, [])
        n = len(h)
        if n < max(2, self.min_obs):
            return False
        mean = sum(h) / n
        var = sum((x - mean) ** 2 for x in h) / (n - 1)
        return mean + self.z * (var / n) ** 0.5 < 0

    def to_dict(self) -> Dict:
        return {"hist": self.hist}

    def load(self, d: Dict) -> None:
        self.hist = d.get("hist", {})
