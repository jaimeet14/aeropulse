"""Player layer: injuries/availability, learned absence impact, and player-prop distributions.

* availability : injury-report status -> probability the player plays
* impact       : how much the team's margin drops when the player is missing (net of the
                 replacement). Learned online from lineups: when a regular is absent and the
                 team under-performs the model, his impact grows. Can be seeded manually.
* stat means   : exponentially weighted mean/variance of each player's box-score stats,
                 used to price props; teammates get a usage boost when a regular is out.
"""
import math
from typing import Dict, List, Optional, Tuple

from .dists import discrete_normal, poisson_pmf

STATUS_AVAIL = {
    "out": 0.0, "suspended": 0.0, "injured": 0.0, "doubtful": 0.25, "questionable": 0.5,
    "game-time decision": 0.5, "day-to-day": 0.6, "probable": 0.85, "available": 1.0, "active": 1.0,
}

# distribution family per stat; anything not listed uses "normal"
STAT_FAMILY = {"goals": "poisson", "assists_fb": "poisson", "shots_on_target": "poisson",
               "aces": "poisson", "double_faults": "poisson", "wickets": "poisson",
               "threes": "poisson", "runs": "exponential"}


def status_availability(status: str) -> float:
    return STATUS_AVAIL.get((status or "").strip().lower(), 1.0)


class PlayerModel:
    def __init__(self, lr: float = 0.01, alpha: float = 0.08, impact_cap: float = 10.0):
        self.lr, self.alpha, self.impact_cap = lr, alpha, impact_cap
        self.impact: Dict[str, float] = {}          # "team|player" -> margin units
        self.apps: Dict[str, float] = {}            # EMA of appearance rate
        self.stats: Dict[str, Dict[str, List[float]]] = {}   # key -> stat -> [mean, var, n]
        self.team_stat: Dict[str, Dict[str, float]] = {}     # team -> stat -> EMA team total

    # ------------------------------------------------------------ availability
    @staticmethod
    def key(team: str, player: str) -> str:
        return f"{team}|{player}"

    def regulars(self, team: str, threshold: float = 0.5) -> List[str]:
        pre = team + "|"
        return [k[len(pre):] for k, v in self.apps.items() if k.startswith(pre) and v >= threshold]

    def availability(self, ev, side: str) -> Dict[str, float]:
        """Probability each known player on `side` plays. Announced lineups beat injury reports."""
        team = ev.home if side == "home" else ev.away
        avail = {p: 1.0 for p in self.regulars(team)}
        for inj in ev.injuries or []:
            if inj.get("team") in (team, side):
                avail[inj["player"]] = status_availability(inj.get("status", ""))
        if ev.lineups and ev.lineups.get(side):
            lineup = set(ev.lineups[side])
            for p in list(avail):
                avail[p] = 1.0 if p in lineup else 0.0
            for p in lineup:
                avail[p] = 1.0
        return avail

    def absence_penalty(self, ev, side: str) -> float:
        team = ev.home if side == "home" else ev.away
        return sum(self.impact.get(self.key(team, p), 0.0) * (1.0 - a)
                   for p, a in self.availability(ev, side).items())

    # ------------------------------------------------------------ learning
    def learn(self, ev, margin_residual: float) -> None:
        """margin_residual = actual home margin - predicted home margin (prediction incl. penalties)."""
        res = ev.result or {}
        pstats = res.get("player_stats") or {}
        for side, sign in (("home", 1.0), ("away", -1.0)):
            team = ev.home if side == "home" else ev.away
            lineups = res.get("lineups") or ev.lineups or {}
            played = set(lineups.get(side) or [])
            if not played and pstats:
                played = {p for p in pstats if self._team_hint(p, team, ev, side)}
            if not played:
                continue
            # absence impact: regulars who did not play
            for p in self.regulars(team):
                if p not in played:
                    k = self.key(team, p)
                    v = self.impact.get(k, 0.0) - self.lr * sign * margin_residual
                    self.impact[k] = max(0.0, min(self.impact_cap, v))
            known = {k for k in self.apps if k.startswith(team + "|")}
            for k in known | {self.key(team, p) for p in played}:
                p = k.split("|", 1)[1]
                prev = self.apps.get(k, 0.6)   # new names start near "regular"
                self.apps[k] = prev + self.alpha * ((1.0 if p in played else 0.0) - prev)
            # box-score stats
            totals: Dict[str, float] = {}
            for p in played:
                for stat, val in (pstats.get(p) or {}).items():
                    self._ema(self.key(team, p), stat, float(val))
                    totals[stat] = totals.get(stat, 0.0) + float(val)
            ts = self.team_stat.setdefault(team, {})
            for stat, val in totals.items():
                ts[stat] = val if stat not in ts else ts[stat] + self.alpha * (val - ts[stat])

    def _team_hint(self, player: str, team: str, ev, side: str) -> bool:
        tag = (ev.result or {}).get("player_team", {}).get(player)
        if tag is not None:
            return tag in (team, side)
        return self.key(team, player) in self.apps

    def _ema(self, k: str, stat: str, x: float) -> None:
        s = self.stats.setdefault(k, {}).get(stat)
        if s is None:
            self.stats[k][stat] = [x, max(1.0, x), 1]
            return
        m, v, n = s
        a = max(self.alpha, 1.0 / (n + 1))
        d = x - m
        m += a * d
        v = (1 - a) * (v + a * d * d)
        self.stats[k][stat] = [m, v, n + 1]

    # ------------------------------------------------------------ props
    def find_team(self, player: str, ev) -> Optional[Tuple[str, str]]:
        for side, team in (("home", ev.home), ("away", ev.away)):
            if self.key(team, player) in self.stats:
                return side, team
        return None

    def prop_dist(self, ev, player: str, stat: str, team_scale: Dict[str, float]) -> Optional[Tuple[Dict[int, float], float]]:
        """(distribution conditional on playing, availability). team_scale[side] scales for matchup."""
        loc = self.find_team(player, ev)
        if not loc:
            return None
        side, team = loc
        s = self.stats.get(self.key(team, player), {}).get(stat)
        if not s or s[2] < 3:
            return None
        avail = self.availability(ev, side)
        mean = s[0] * team_scale.get(side, 1.0) * self._usage_boost(team, stat, avail, player)
        fam = STAT_FAMILY.get(stat, "normal")
        if fam == "poisson":
            hi = int(mean + 8 * math.sqrt(mean + 1)) + 2
            dist = {k: poisson_pmf(k, max(mean, 1e-6)) for k in range(hi + 1)}
        elif fam == "exponential":  # batting runs: heavy tail, many low scores
            lam = 1.0 / max(mean, 1.0)
            hi = int(mean * 8) + 10
            dist = {k: math.exp(-lam * k) - math.exp(-lam * (k + 1)) for k in range(hi)}
        else:
            sd = math.sqrt(max(s[1], 1.0))
            dist = discrete_normal(mean, sd, 0, int(mean + 6 * sd) + 1)
        z = sum(dist.values())
        return {k: v / z for k, v in dist.items()}, avail.get(player, 1.0)

    def _usage_boost(self, team: str, stat: str, avail: Dict[str, float], player: str) -> float:
        team_avg = self.team_stat.get(team, {}).get(stat)
        if not team_avg:
            return 1.0
        expected = sum(self.stats.get(self.key(team, p), {}).get(stat, [0.0])[0] * a
                       for p, a in avail.items() if p != player)
        expected += self.stats.get(self.key(team, player), {}).get(stat, [0.0])[0]
        if expected <= 0:
            return 1.0
        return max(1.0, min(1.25, team_avg / expected))

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> Dict:
        return {"impact": self.impact, "apps": self.apps, "stats": self.stats, "team_stat": self.team_stat}

    def load(self, d: Dict) -> None:
        self.impact, self.apps = d.get("impact", {}), d.get("apps", {})
        self.stats, self.team_stat = d.get("stats", {}), d.get("team_stat", {})
