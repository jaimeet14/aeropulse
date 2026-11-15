"""Per-sport self-learning engines.

Every engine follows the same loop:
    forecast(event)  -> Forecast over every market (uses ratings + context factors + injuries)
    update(event)    -> learn from the settled result (ratings, factor weights, player impact & stats)

What each one learns:
  football   : attack/defence per team (Poisson goals), home advantage, scoring level
  basketball : offensive/defensive points ratings, home edge, margin & total volatility
  cricket    : batting/bowling runs ratings per format, venue scoring level, toss value
  tennis     : overall + surface Elo, per-player serve dominance (drives total games)
plus, in all sports: context-factor weights and player absence impact / prop stats.
"""
import math
from typing import Dict

from .dists import discrete_normal, match_distribution, score_matrix, serve_probs_for_win_prob
from .factors import FactorModel, clip, num
from .markets import Forecast
from .players import PlayerModel, status_availability


class Engine:
    sport = ""
    margin_priors: Dict[str, float] = {}
    total_priors: Dict[str, float] = {}

    def __init__(self):
        self.players = PlayerModel(lr=self.impact_lr)
        self.fm = FactorModel(self.margin_priors, lr=self.factor_lr)
        self.ft = FactorModel(self.total_priors, lr=self.factor_lr)
        self.n: Dict[str, int] = {}

    impact_lr = 0.01
    factor_lr = 0.002

    # -- shared helpers -------------------------------------------------------
    def features(self, ev) -> Dict[str, float]:
        c = ev.context or {}
        x = {
            "rest_diff": clip(num(c, "rest_home", 2) - num(c, "rest_away", 2), -3, 3),
            "b2b_home": float(num(c, "rest_home", 2) <= 1),
            "b2b_away": float(num(c, "rest_away", 2) <= 1),
            "travel_diff": clip((num(c, "travel_km_away") - num(c, "travel_km_home")) / 1000, -5, 5),
            "rain": num(c, "rain"),
            "wind": num(c, "wind_ms") / 10,
            "altitude": num(c, "altitude_km"),
        }
        return {k: v for k, v in x.items() if v}

    def penalties(self, ev):
        return self.players.absence_penalty(ev, "home"), self.players.absence_penalty(ev, "away")

    def confident(self, ev, min_games: int) -> bool:
        return self.n.get(ev.home, 0) >= min_games and self.n.get(ev.away, 0) >= min_games

    def _count(self, ev):
        for t in (ev.home, ev.away):
            self.n[t] = self.n.get(t, 0) + 1

    def _props(self, ev, scale):
        return lambda player, stat: self.players.prop_dist(ev, player, stat, scale)

    def _team_scale(self, ev, stat, exp_home, exp_away):
        out = {}
        for side, team, e in (("home", ev.home, exp_home), ("away", ev.away, exp_away)):
            avg = self.players.team_stat.get(team, {}).get(stat)
            out[side] = clip(e / avg, 0.7, 1.4) if avg else 1.0
        return out

    # -- persistence ------------------------------------------------------------
    def state(self) -> Dict:
        return {}

    def load_state(self, d: Dict) -> None:
        pass

    def to_dict(self) -> Dict:
        return {"sport": self.sport, "n": self.n, "players": self.players.to_dict(),
                "fm": self.fm.to_dict(), "ft": self.ft.to_dict(), "state": self.state()}

    def load(self, d: Dict) -> "Engine":
        self.n = d.get("n", {})
        self.players.load(d.get("players", {}))
        self.fm.load(d.get("fm", {}))
        self.ft.load(d.get("ft", {}))
        self.load_state(d.get("state", {}))
        return self


# ============================================================================ football
class FootballEngine(Engine):
    sport = "football"
    margin_priors = {"rest_diff": 0.03, "travel_diff": 0.02}
    total_priors = {"rain": -0.10, "wind": -0.15}
    impact_lr = 0.004
    factor_lr = 0.001

    def __init__(self, lr: float = 0.04):
        super().__init__()
        self.lr = lr
        self.mu, self.home, self.rho = math.log(1.35), 0.25, -0.05
        self.att: Dict[str, float] = {}
        self.dfn: Dict[str, float] = {}

    def _lams(self, ev):
        neutral = bool((ev.context or {}).get("neutral"))
        la = math.exp(self.mu + (0 if neutral else self.home) + self.att.get(ev.home, 0) - self.dfn.get(ev.away, 0))
        lb = math.exp(self.mu + self.att.get(ev.away, 0) - self.dfn.get(ev.home, 0))
        x = self.features(ev)
        ph, pa = self.penalties(ev)
        dm = self.fm.adjust(x) - ph + pa
        dt = self.ft.adjust(x)
        la, lb = max(0.05, la + dm / 2), max(0.05, lb - dm / 2)
        s = la + lb
        k = max(0.3, (s + dt) / s)
        return la * k, lb * k, x

    def forecast(self, ev) -> Forecast:
        la, lb, _ = self._lams(ev)
        mat = score_matrix(la, lb, self.rho)
        m = {q: {} for q in ("winner", "margin", "total", "home", "away", "btts", "score")}
        for i, row in enumerate(mat):
            for j, p in enumerate(row):
                for q, v in (("winner", (i > j) - (i < j)), ("margin", i - j), ("total", i + j),
                             ("home", i), ("away", j), ("btts", i > 0 and j > 0), ("score", f"{i}-{j}")):
                    m[q][v] = m[q].get(v, 0.0) + p
        scale = self._team_scale(ev, "goals", la, lb)
        return Forecast(m, {"lam_home": la, "lam_away": lb}, self._props(ev, scale))

    def update(self, ev) -> None:
        gh, ga = ev.result["home"], ev.result["away"]
        la, lb, x = self._lams(ev)
        eh, ea = gh - la, ga - lb
        h, a = ev.home, ev.away
        self.att[h] = self.att.get(h, 0) + self.lr * eh
        self.dfn[a] = self.dfn.get(a, 0) - self.lr * eh
        self.att[a] = self.att.get(a, 0) + self.lr * ea
        self.dfn[h] = self.dfn.get(h, 0) - self.lr * ea
        self.mu += self.lr * 0.02 * (eh + ea) / 2
        if not (ev.context or {}).get("neutral"):
            self.home += self.lr * 0.02 * (eh - ea)
        mres = (gh - ga) - (la - lb)
        self.fm.learn(x, mres)
        self.ft.learn(x, (gh + ga) - (la + lb))
        self.players.learn(ev, mres)
        self._count(ev)

    def state(self):
        return {"mu": self.mu, "home": self.home, "rho": self.rho, "att": self.att, "dfn": self.dfn}

    def load_state(self, d):
        self.__dict__.update({k: v for k, v in d.items() if k in ("mu", "home", "rho", "att", "dfn")})


# ============================================================================ points sports
class _PointsEngine(Engine):
    """Shared by basketball and cricket: team scores from offence/defence ratings + normal noise."""
    base0, home0, sdm0, sdt0, lr0, lo_hi = 110.0, 2.5, 13.0, 19.0, 0.06, (0, 250)
    score_stat = "points"

    def __init__(self, lr: float = None):
        super().__init__()
        self.lr = lr or self.lr0
        self.base: Dict[str, float] = {}       # per format
        self.home, self.sdm, self.sdt = self.home0, {}, {}
        self.off: Dict[str, float] = {}
        self.dfn: Dict[str, float] = {}
        self.venue: Dict[str, float] = {}

    def fmt(self, ev) -> str:
        return ""

    def defaults(self, fmt):
        return self.base0, self.sdm0, self.sdt0

    def _k(self, team, fmt):
        return f"{team}|{fmt}" if fmt else team

    def _means(self, ev):
        f = self.fmt(ev)
        b0, m0, t0 = self.defaults(f)
        base = self.base.get(f, b0)
        neutral = bool((ev.context or {}).get("neutral"))
        hadv = 0.0 if neutral else self.home * (b0 / self.base0)
        hp = base + hadv / 2 + self.off.get(self._k(ev.home, f), 0) - self.dfn.get(self._k(ev.away, f), 0)
        ap = base - hadv / 2 + self.off.get(self._k(ev.away, f), 0) - self.dfn.get(self._k(ev.home, f), 0)
        x = self.features(ev)
        ph, pa = self.penalties(ev)
        venue = (ev.context or {}).get("venue", "")
        mu_m = hp - ap + self.fm.adjust(x) - ph + pa
        mu_t = hp + ap + self.ft.adjust(x) + self.venue.get(venue, 0.0)
        return mu_m, mu_t, self.sdm.get(f, m0), self.sdt.get(f, t0), x, f

    def forecast(self, ev) -> Forecast:
        mu_m, mu_t, sdm, sdt, _, _ = self._means(ev)
        lo, hi = self.lo_hi
        margin = discrete_normal(mu_m, sdm, int(mu_m - 6 * sdm), int(mu_m + 6 * sdm))
        p0 = margin.get(0, 0.0)
        ph = sum(p for k, p in margin.items() if k > 0) + p0 / 2      # ties -> overtime / super over
        sd_team = math.sqrt(sdm**2 + sdt**2) / 2
        e_h, e_a = (mu_t + mu_m) / 2, (mu_t - mu_m) / 2
        m = {
            "winner": {1: ph, -1: 1 - ph},
            "margin": margin,
            "total": discrete_normal(mu_t, sdt, max(lo, int(mu_t - 6 * sdt)), int(mu_t + 6 * sdt)),
            "home": discrete_normal(e_h, sd_team, max(lo, int(e_h - 6 * sd_team)), int(e_h + 6 * sd_team)),
            "away": discrete_normal(e_a, sd_team, max(lo, int(e_a - 6 * sd_team)), int(e_a + 6 * sd_team)),
        }
        scale = self._team_scale(ev, self.score_stat, e_h, e_a)
        return Forecast(m, {"exp_home": e_h, "exp_away": e_a, "sd_margin": sdm, "sd_total": sdt},
                        self._props(ev, scale))

    def update(self, ev) -> None:
        H, A = ev.result["home"], ev.result["away"]
        mu_m, mu_t, sdm, sdt, x, f = self._means(ev)
        b0, _, _ = self.defaults(f)
        eh, ea = H - (mu_t + mu_m) / 2, A - (mu_t - mu_m) / 2
        hk, ak = self._k(ev.home, f), self._k(ev.away, f)
        lr = self.lr
        self.off[hk] = self.off.get(hk, 0) + lr * eh
        self.dfn[ak] = self.dfn.get(ak, 0) - lr * eh
        self.off[ak] = self.off.get(ak, 0) + lr * ea
        self.dfn[hk] = self.dfn.get(hk, 0) - lr * ea
        self.base[f] = self.base.get(f, b0) + lr * 0.05 * (eh + ea) / 2
        if not (ev.context or {}).get("neutral"):
            self.home += lr * 0.05 * (eh - ea) * (self.base0 / b0)
        mres, tres = (H - A) - mu_m, (H + A) - mu_t
        self.sdm[f] = math.sqrt(sdm**2 + 0.01 * (mres**2 - sdm**2))
        self.sdt[f] = math.sqrt(sdt**2 + 0.01 * (tres**2 - sdt**2))
        venue = (ev.context or {}).get("venue")
        if venue:
            self.venue[venue] = self.venue.get(venue, 0.0) + 0.05 * tres
        self.fm.learn(x, mres)
        self.ft.learn(x, tres)
        self.players.learn(ev, mres)
        self._count(ev)

    def state(self):
        return {k: getattr(self, k) for k in ("base", "home", "sdm", "sdt", "off", "dfn", "venue")}

    def load_state(self, d):
        self.__dict__.update({k: v for k, v in d.items() if k in ("base", "home", "sdm", "sdt", "off", "dfn", "venue")})


class BasketballEngine(_PointsEngine):
    sport = "basketball"
    base0, home0, sdm0, sdt0, lr0, lo_hi = 113.0, 2.5, 13.0, 19.0, 0.06, (40, 400)
    margin_priors = {"rest_diff": 0.6, "b2b_home": -1.5, "b2b_away": 1.5, "travel_diff": 0.3, "altitude": 1.0}
    total_priors = {"b2b_home": -1.0, "b2b_away": -1.0}
    impact_lr = 0.03


class CricketEngine(_PointsEngine):
    """Limited-overs cricket (T20 / ODI). Test matches (draws, two innings) are not modelled."""
    sport = "cricket"
    base0, home0, sdm0, sdt0, lr0, lo_hi = 160.0, 8.0, 32.0, 40.0, 0.08, (0, 900)
    score_stat = "runs"
    margin_priors = {"toss": 3.0, "rest_diff": 0.5, "travel_diff": 0.5}
    total_priors = {"rain": -15.0, "dew": 8.0}
    impact_lr = 0.05
    FORMATS = {"t20": (160.0, 32.0, 40.0), "odi": (265.0, 50.0, 60.0)}

    def fmt(self, ev):
        f = str((ev.context or {}).get("format", "t20")).lower()
        return f if f in self.FORMATS else "t20"

    def defaults(self, fmt):
        return self.FORMATS[fmt]

    def features(self, ev):
        x = super().features(ev)
        c = ev.context or {}
        toss = c.get("toss")
        if toss in ("home", ev.home):
            x["toss"] = 1.0
        elif toss in ("away", ev.away):
            x["toss"] = -1.0
        if num(c, "dew"):
            x["dew"] = num(c, "dew")
        return x


# ============================================================================ tennis
class TennisEngine(Engine):
    """`home`/`away` are player A / player B. Neutral venue assumed."""
    sport = "tennis"
    margin_priors = {"injury_home": -150.0, "injury_away": 150.0, "fatigue_diff": -15.0}
    total_priors = {}
    SURFACE_SERVE = {"hard": 0.640, "clay": 0.600, "grass": 0.670, "indoor": 0.655}
    factor_lr = 0.5

    def __init__(self):
        super().__init__()
        self.elo: Dict[str, float] = {}
        self.elo_s: Dict[str, float] = {}
        self.serve: Dict[str, float] = {}      # learned serve dominance (adds to hold rates)

    def features(self, ev):
        c = ev.context or {}
        x = {"fatigue_diff": clip(num(c, "sets_last_3d_home") - num(c, "sets_last_3d_away"), -6, 6)}
        for inj in ev.injuries or []:
            for side, name in (("home", ev.home), ("away", ev.away)):
                if inj.get("player") == name or inj.get("team") == side:
                    x[f"injury_{side}"] = 1.0 - status_availability(inj.get("status", ""))
        return {k: v for k, v in x.items() if v}

    def _p(self, ev):
        surf = str((ev.context or {}).get("surface", "hard")).lower()
        def r(p):
            return 0.5 * self.elo.get(p, 1500.0) + 0.5 * self.elo_s.get(f"{p}|{surf}", 1500.0)
        x = self.features(ev)
        diff = r(ev.home) - r(ev.away) + self.fm.adjust(x)
        return 1.0 / (1.0 + 10 ** (-diff / 400)), surf, x

    def forecast(self, ev) -> Forecast:
        p, surf, _ = self._p(ev)
        best_of = int(num(ev.context or {}, "best_of", 3))
        base = self.SURFACE_SERVE.get(surf, 0.64) + self.serve.get(ev.home, 0) + self.serve.get(ev.away, 0)
        base = clip(base, 0.5, 0.78)
        pa, pb = serve_probs_for_win_prob(p, best_of, base)
        joint = match_distribution(pa, pb, best_of)
        m = {q: {} for q in ("winner", "set_margin", "score", "margin", "total", "home", "away")}
        for (sa, sb, ga, gb), pr in joint.items():
            for q, v in (("winner", 1 if sa > sb else -1), ("set_margin", sa - sb), ("score", f"{sa}-{sb}"),
                         ("margin", ga - gb), ("total", ga + gb), ("home", ga), ("away", gb)):
                m[q][v] = m[q].get(v, 0.0) + pr
        return Forecast(m, {"p_home": p, "serve_home": pa, "serve_away": pb}, self._props(ev, {}))

    def update(self, ev) -> None:
        p, surf, x = self._p(ev)
        y = 1.0 if ev.result["home"] > ev.result["away"] else 0.0
        for pl, sign in ((ev.home, 1), (ev.away, -1)):
            k = 250.0 / (self.n.get(pl, 0) + 5) ** 0.4
            d = sign * k * (y - p)
            self.elo[pl] = self.elo.get(pl, 1500.0) + d
            ks = f"{pl}|{surf}"
            self.elo_s[ks] = self.elo_s.get(ks, 1500.0) + d
        self.fm.learn(x, (y - p))
        sets = ev.result.get("sets")
        if sets:
            exp_total = sum(k * v for k, v in self.forecast(ev).m["total"].items())
            res = sum(a + b for a, b in sets) - exp_total
            for pl in (ev.home, ev.away):
                self.serve[pl] = clip(self.serve.get(pl, 0.0) + 0.0005 * res, -0.08, 0.08)
        self.players.learn(ev, 0.0)
        self._count(ev)

    def state(self):
        return {"elo": self.elo, "elo_s": self.elo_s, "serve": self.serve}

    def load_state(self, d):
        self.__dict__.update({k: v for k, v in d.items() if k in ("elo", "elo_s", "serve")})


ENGINES = {"football": FootballEngine, "basketball": BasketballEngine,
           "cricket": CricketEngine, "tennis": TennisEngine}


def make_engine(sport: str) -> Engine:
    try:
        return ENGINES[sport.lower()]()
    except KeyError:
        raise ValueError(f"unknown sport {sport!r}; choose from {sorted(ENGINES)}")


def engine_from_dict(d: Dict) -> Engine:
    return make_engine(d["sport"]).load(d)
