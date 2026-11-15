"""Walk-forward backtest over every offered market: forecast -> pick -> settle -> learn.

Events are processed in date order and the engine learns from a result only after its bets
are settled, so nothing from the future leaks into a decision.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List

from .data import Event
from .decide import _group_key, evaluate, market_probs
from .engines import Engine
from .markets import settle
from .risk import RiskConfig, RiskManager, TrustGate


@dataclass
class MarketStats:
    bets: int = 0
    wins: int = 0
    staked: float = 0.0
    profit: float = 0.0


@dataclass
class Report:
    start_bankroll: float
    end_bankroll: float = 0.0
    events: int = 0
    max_drawdown: float = 0.0
    halted: bool = False
    by_market: Dict[str, MarketStats] = field(default_factory=dict)
    model_ll: float = 0.0
    market_ll: float = 0.0
    ll_n: int = 0
    clv_sum: float = 0.0
    clv_n: int = 0
    curve: List[float] = field(default_factory=list)

    @property
    def bets(self):
        return sum(m.bets for m in self.by_market.values())

    @property
    def staked(self):
        return sum(m.staked for m in self.by_market.values())

    def summary(self) -> str:
        g = self.end_bankroll / self.start_bankroll - 1
        y = (self.end_bankroll - self.start_bankroll) / self.staked if self.staked else 0.0
        lines = [
            f"events processed      : {self.events}",
            f"bets placed           : {self.bets}",
            f"bankroll              : {self.start_bankroll:.2f} -> {self.end_bankroll:.2f} ({g:+.1%})",
            f"yield (profit/staked) : {y:+.2%}",
            f"max drawdown          : {self.max_drawdown:.1%}",
            f"kill switch tripped   : {self.halted}",
        ]
        if self.ll_n:
            lines.append(f"winner log-loss       : model {self.model_ll / self.ll_n:.4f} vs market "
                         f"{self.market_ll / self.ll_n:.4f} over {self.ll_n} events (lower is better)")
        lines.append(f"avg closing-line value: {self.clv_sum / self.clv_n:+.2%} over {self.clv_n} bets"
                     if self.clv_n else "avg closing-line value: n/a (no closing odds)")
        gate = getattr(self, "gate", None)
        if gate and gate.hist:
            lines.append("")
            lines.append("trust gate (model Brier - market Brier, negative = model better):")
            for mk in sorted(gate.hist):
                sc, n = gate.score(mk)
                lines.append(f"  {mk:<20}{sc:+.5f} over {n:>3} obs -> {'BETTING' if gate.allows(mk) else 'blocked'}")
        lines.append("")
        lines.append(f"{'market':<22}{'bets':>6}{'won':>6}{'staked':>11}{'profit':>11}{'yield':>9}")
        for k, m in sorted(self.by_market.items(), key=lambda kv: -kv[1].staked):
            yy = m.profit / m.staked if m.staked else 0.0
            lines.append(f"{k:<22}{m.bets:>6}{m.wins:>6}{m.staked:>11.2f}{m.profit:>+11.2f}{yy:>+9.1%}")
        return "\n".join(lines)


WINNER_MARKETS = {"1x2": {"home": 1, "draw": 0, "away": -1}, "moneyline": {"home": 1, "away": -1}}


def run_backtest(events: List[Event], engine: Engine, bankroll: float = 1000.0,
                 risk: RiskConfig = None, gate: TrustGate = None) -> Report:
    risk = risk or RiskConfig()
    gate = gate if gate is not None else TrustGate()
    rm = RiskManager(risk, bankroll)
    rep = Report(start_bankroll=bankroll)
    peak = bankroll
    for i, ev in enumerate(events):
        if not ev.result:
            continue
        eid = ev.id or f"{ev.date}:{ev.home}:{ev.away}:{i}"
        if engine.confident(ev, risk.min_games):
            fc = engine.forecast(ev)
            _score_winner(rep, fc, ev)
            stake_base = bankroll
            picks = evaluate(fc, ev, risk)
            for pk in picks:
                if pk.reason or not gate.allows(pk.odds.market):
                    continue
                stake = rm.stake(pk.p_win, pk.p_push, pk.odds.price, stake_base, ev.date, eid)
                if stake <= 0:
                    continue
                o = pk.odds
                outcome = settle(ev.sport, o.market, o.selection, o.line, ev.result)
                if outcome == "void":
                    continue
                pnl = stake * (o.price - 1.0) if outcome == "win" else 0.0 if outcome == "push" else -stake
                bankroll += pnl
                key = o.market if not o.market.startswith("player_") else o.market
                ms = rep.by_market.setdefault(key, MarketStats())
                ms.bets += 1
                ms.wins += outcome == "win"
                ms.staked += stake
                ms.profit += pnl
                if o.closing:
                    rep.clv_sum += o.price / o.closing - 1.0
                    rep.clv_n += 1
            seen = set()
            for pk in picks:  # self-assessment: one observation per market line, bet or not
                o = pk.odds
                if _group_key(o) in seen:
                    continue
                seen.add(_group_key(o))
                out = settle(ev.sport, o.market, o.selection, o.line, ev.result)
                if out in ("win", "lose") and pk.p_push < 0.5:
                    gate.record(o.market, pk.p_win / (1 - pk.p_push), pk.market_p, out == "win")
        engine.update(ev)
        rm.observe(bankroll)
        peak = max(peak, bankroll)
        rep.max_drawdown = max(rep.max_drawdown, 1.0 - bankroll / peak)
        rep.curve.append(bankroll)
        rep.events += 1
    rep.end_bankroll = bankroll
    rep.halted = rm.halted
    rep.gate = gate
    return rep


def _score_winner(rep: Report, fc, ev: Event) -> None:
    for mk, sels in WINNER_MARKETS.items():
        offered = [o for o in ev.odds if o.market == mk and o.selection in sels]
        if len(offered) != len(sels):
            continue
        h, a = ev.result["home"], ev.result["away"]
        actual = (h > a) - (h < a)
        win = [o for o in offered if sels[o.selection] == actual]
        if not win:
            return
        mp = market_probs(offered)
        p_model = fc.m["winner"].get(actual, 0.0)
        rep.model_ll += -math.log(max(p_model, 1e-9))
        rep.market_ll += -math.log(max(mp[id(win[0])], 1e-9))
        rep.ll_n += 1
        return
