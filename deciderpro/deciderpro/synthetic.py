"""Synthetic multi-sport generator for testing the machinery end to end.

IMPORTANT: profits on synthetic data say nothing about real-world results. A hidden "true" engine
generates results; the simulated bookmaker sees noisy ratings and only part of the injury news
(`book_injury_info`), then adds a margin. Closing odds use better information, so CLV is meaningful.
Set book_noise=0 and book_injury_info=1 to get a sharp book that should not be beatable.
"""
import copy
import math
import random
from datetime import date, timedelta
from typing import Dict, List

from .data import Event, Odds
from .dists import set_distribution
from .engines import make_engine

ROSTER = {"football": 11, "basketball": 8, "cricket": 11, "tennis": 0}
PROP_STAT = {"football": "goals", "basketball": "points", "cricket": "runs"}


def _sample(rng, dist: Dict) -> object:
    r, acc = rng.random(), 0.0
    for k, p in dist.items():
        acc += p
        if r < acc:
            return k
    return k


def _poisson(rng, lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p < L:
            return k
        k += 1


def _price(p, margin, cap=0.97):
    return round(1.0 / min(max(p, 0.005) * (1 + margin), cap), 3)


class World:
    def __init__(self, sport: str, n_teams: int, rng: random.Random):
        self.sport, self.rng = sport, rng
        self.truth = make_engine(sport)
        self.teams = [f"{sport[:3].upper()}{i:02d}" for i in range(n_teams)]
        self.roster: Dict[str, List[str]] = {}
        self.share: Dict[str, float] = {}
        t = self.truth
        for team in self.teams:
            if sport == "football":
                t.att[team], t.dfn[team] = rng.gauss(0, 0.25), rng.gauss(0, 0.25)
            elif sport in ("basketball", "cricket"):
                k = f"{team}|t20" if sport == "cricket" else team
                sd = 4.0 if sport == "basketball" else 10.0
                t.off[k], t.dfn[k] = rng.gauss(0, sd), rng.gauss(0, sd)
            else:
                t.elo[team] = rng.gauss(1500, 160)
                for s in ("hard", "clay", "grass"):
                    t.elo_s[f"{team}|{s}"] = t.elo[team] + rng.gauss(0, 70)
                t.serve[team] = rng.gauss(0, 0.015)
            n = ROSTER[sport]
            if not n:
                continue
            names = [f"{team}_P{j}" for j in range(n)]
            self.roster[team] = names
            w = sorted((rng.random() ** 2 for _ in names), reverse=True)
            tot = sum(w)
            unit = {"football": 0.45, "basketball": 7.0, "cricket": 14.0}[sport]
            for nm, x in zip(names, w):
                self.share[nm] = x / tot
                key = t.players.key(team, nm)
                t.players.apps[key] = 1.0
                t.players.impact[key] = unit * x / tot * n / 3
        # truth context effects are a bit stronger than the learner's priors
        for fm in (t.fm, t.ft):
            for k in fm.w:
                fm.w[k] *= 1.4

    def drift(self, team):
        t, r = self.truth, self.rng
        if self.sport == "football":
            t.att[team] += r.gauss(0, 0.01)
        elif self.sport == "basketball":
            t.off[team] += r.gauss(0, 0.15)
        elif self.sport == "cricket":
            t.off[f"{team}|t20"] += r.gauss(0, 0.4)
        else:
            t.elo[team] += r.gauss(0, 4)


def generate(sport: str, n_events: int = 2000, n_teams: int = 20, seed: int = 7, margin: float = 0.05,
             book_noise: float = 1.0, book_injury_info: float = 0.4, injury_rate: float = 0.06) -> List[Event]:
    rng = random.Random(seed)
    w = World(sport, n_teams, rng)
    start = date(2022, 1, 1)
    per_day = max(1, n_teams // 4)
    events: List[Event] = []
    for i in range(n_events):
        h, a = rng.sample(w.teams, 2)
        ev = Event(sport, (start + timedelta(days=i // per_day)).isoformat(), h, a, id=f"{sport}-{i}")
        ev.context = _context(sport, rng)
        played = _injuries(w, ev, rng, injury_rate)
        actual = copy.copy(ev)
        actual.lineups = played
        tf = w.truth.forecast(actual)
        ev.result = _result(w, ev, tf, played, rng)
        ev.odds = _odds(w, ev, tf, played, rng, margin, book_noise, book_injury_info)
        events.append(ev)
        for t in (h, a):
            w.drift(t)
    return events


def _context(sport, rng):
    c = {"rest_home": rng.choice([1, 2, 2, 3, 4]), "rest_away": rng.choice([1, 2, 2, 3, 4])}
    if sport == "tennis":
        c = {"surface": rng.choice(["hard", "hard", "clay", "grass"]), "best_of": rng.choice([3, 3, 5])}
    if sport == "cricket":
        c.update(format="t20", toss=rng.choice(["home", "away"]), venue=f"V{rng.randrange(6)}")
    if sport == "football" and rng.random() < 0.2:
        c["rain"] = 1
    return c


def _injuries(w, ev, rng, rate):
    played = {}
    for side, team in (("home", ev.home), ("away", ev.away)):
        if w.sport == "tennis":
            if rng.random() < rate / 2:
                ev.injuries.append({"player": team, "team": side, "status": "questionable"})
            continue
        lineup = []
        for p in w.roster[team]:
            if rng.random() < rate:
                status = rng.choice(["out", "out", "doubtful", "questionable", "questionable"])
                ev.injuries.append({"player": p, "team": team, "status": status})
                plays = rng.random() < {"out": 0.0, "doubtful": 0.25, "questionable": 0.5}[status]
            else:
                plays = True
            if plays:
                lineup.append(p)
        played[side] = lineup
    return played


def _result(w, ev, tf, played, rng):
    sport = w.sport
    if sport == "football":
        H, A = _poisson(rng, tf.meta["lam_home"]), _poisson(rng, tf.meta["lam_away"])
    elif sport in ("basketball", "cricket"):
        mm = (tf.meta["exp_home"] - tf.meta["exp_away"])
        tt = tf.meta["exp_home"] + tf.meta["exp_away"]
        m, t = rng.gauss(mm, tf.meta["sd_margin"]), rng.gauss(tt, tf.meta["sd_total"])
        H, A = max(0, round((t + m) / 2)), max(0, round((t - m) / 2))
        if H == A:
            H, A = (H + 3, A) if rng.random() < 0.5 else (H, A + 3)
    else:
        need = int(ev.context.get("best_of", 3)) // 2 + 1
        sd = set_distribution(tf.meta["serve_home"], tf.meta["serve_away"])
        sets, sh, sa = [], 0, 0
        while max(sh, sa) < need:
            g = _sample(rng, sd)
            sets.append(list(g))
            sh, sa = sh + (g[0] > g[1]), sa + (g[1] > g[0])
        return {"home": sh, "away": sa, "sets": sets}
    stats, pteam = {}, {}
    stat = PROP_STAT[sport]
    for side, total in (("home", H), ("away", A)):
        team = ev.home if side == "home" else ev.away
        ps = played[side]
        z = sum(w.share[p] for p in ps) or 1.0
        for p in ps:
            pteam[p] = team
            exp = total * w.share[p] / z
            if sport == "football":
                val = 0
            elif sport == "basketball":
                val = max(0, round(rng.gauss(exp, 0.25 * exp + 1.5)))
            else:
                val = int(rng.expovariate(1.0 / max(exp, 0.5)))
            stats[p] = {stat: val}
        if sport == "football":
            for _ in range(total):
                if ps:
                    scorer = _sample(rng, {p: w.share[p] / z for p in ps})
                    stats[scorer][stat] += 1
    return {"home": H, "away": A, "lineups": played, "player_stats": stats, "player_team": pteam}


def _book(w, ev, rng, noise, info):
    book = copy.deepcopy(w.truth)
    t = book
    for team in (ev.home, ev.away):
        if w.sport == "football":
            t.att[team] += rng.gauss(0, 0.12 * noise)
        elif w.sport == "basketball":
            t.off[team] += rng.gauss(0, 2.0 * noise)
        elif w.sport == "cricket":
            t.off[f"{team}|t20"] += rng.gauss(0, 6.0 * noise)
        else:
            t.elo[team] += rng.gauss(0, 50 * noise)
    seen = copy.copy(ev)
    seen.injuries = [x for x in ev.injuries if rng.random() < info]
    seen.result = None
    return book.forecast(seen)


def _odds(w, ev, tf, played, rng, margin, noise, info):
    opening = _book(w, ev, rng, noise, info)
    closing = _book(w, ev, rng, noise * 0.4, min(1.0, info + 0.5))
    offers = []

    def add(market, sel, line=None):
        po, pc = opening.prob(market, sel, line), closing.prob(market, sel, line)
        if po and pc and po[0] > 0:
            offers.append(Odds(market, sel, _price(po[0], margin), line, _price(pc[0], margin)))

    def median_line(dist, mean):
        return math.floor(mean) + 0.5

    s = w.sport
    if s == "football":
        for sel in ("home", "draw", "away"):
            add("1x2", sel)
        for line in (1.5, 2.5, 3.5):
            add("total", "over", line), add("total", "under", line)
        mm = sum(k * p for k, p in opening.m["margin"].items())
        L = -(math.floor(mm) + 0.5)
        add("spread", "home", L), add("spread", "away", -L)
        add("btts", "yes"), add("btts", "no")
    else:
        add("moneyline", "home"), add("moneyline", "away")
        mm = sum(k * p for k, p in opening.m["margin"].items())
        tt = sum(k * p for k, p in opening.m["total"].items())
        L = -(math.floor(mm) + 0.5)
        add("spread", "home", L), add("spread", "away", -L)
        T = math.floor(tt) + 0.5
        add("total", "over", T), add("total", "under", T)
        if s in ("basketball", "cricket"):
            for q, mk in (("home", "team_total_home"), ("away", "team_total_away")):
                e = sum(k * p for k, p in opening.m[q].items())
                add(mk, "over", math.floor(e) + 0.5), add(mk, "under", math.floor(e) + 0.5)
        if s == "tennis":
            for sc in opening.m["score"]:
                add("set_betting", sc)
            add("set_handicap", "home", -1.5), add("set_handicap", "away", 1.5)
    # player props on the top players who are listed in the lineup
    if s in PROP_STAT:
        stat = PROP_STAT[s]
        for side, team in (("home", ev.home), ("away", ev.away)):
            top = sorted(w.roster[team], key=lambda p: -w.share[p])[:3 if s == "basketball" else 2]
            for p in top:
                if p not in played[side]:
                    continue
                exp = _prop_mean(tf, w, p, side, stat, played)
                line = 0.5 if s == "football" else math.floor(exp) + 0.5
                fake_po = _prop_prob(s, exp * math.exp(rng.gauss(0, 0.06 * noise)), line)
                fake_pc = _prop_prob(s, exp * math.exp(rng.gauss(0, 0.025 * noise)), line)
                for sel, po, pc in ((f"{p} over", fake_po, fake_pc), (f"{p} under", 1 - fake_po, 1 - fake_pc)):
                    if s == "football" and sel.endswith("under"):
                        continue
                    offers.append(Odds(f"player_{stat}", sel, _price(po, margin), line, _price(pc, margin)))
    return offers


def _prop_mean(tf, w, p, side, stat, played):
    z = sum(w.share[x] for x in played[side]) or 1.0
    team_exp = tf.meta.get("lam_home" if side == "home" else "lam_away") or \
        tf.meta.get("exp_home" if side == "home" else "exp_away")
    return team_exp * w.share[p] / z


def _prop_prob(sport, mean, line):
    if sport == "football":
        return 1 - math.exp(-mean)
    if sport == "cricket":
        return math.exp(-(line + 0.5) / max(mean, 0.5))
    sd = 0.25 * mean + 1.5
    return 0.5 * (1 - math.erf((line - mean) / (sd * math.sqrt(2))))
