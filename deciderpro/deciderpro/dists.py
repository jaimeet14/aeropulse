"""Probability distributions that turn ratings into prices for every market.

football   : bivariate Poisson scoreline matrix with Dixon-Coles low-score correction
basketball : normal approximations for margin and total, discretised to whole points
tennis     : point -> game -> tiebreak -> set -> match Markov chain (exact)
cricket    : normal approximation of each innings' runs
"""
import math
from functools import lru_cache
from typing import Dict, List, Tuple


# ---------------------------------------------------------------- helpers
def norm_cdf(x: float, mu: float = 0.0, sd: float = 1.0) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sd * math.sqrt(2.0))))


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) if lam > 0 else float(k == 0)


# ---------------------------------------------------------------- football
def score_matrix(lam_h: float, lam_a: float, rho: float = -0.05, max_goals: int = 10) -> List[List[float]]:
    """P(home=i, away=j). rho<0 inflates 0-0/1-1 and deflates 1-0/0-1 (Dixon & Coles 1997)."""
    ph = [poisson_pmf(i, lam_h) for i in range(max_goals + 1)]
    pa = [poisson_pmf(j, lam_a) for j in range(max_goals + 1)]
    m = [[ph[i] * pa[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]
    m[0][0] *= max(0.0, 1 - lam_h * lam_a * rho)
    m[0][1] *= max(0.0, 1 + lam_h * rho)
    m[1][0] *= max(0.0, 1 + lam_a * rho)
    m[1][1] *= max(0.0, 1 - rho)
    s = sum(map(sum, m))
    return [[x / s for x in row] for row in m]


# ---------------------------------------------------------------- integer-valued normal
def discrete_normal(mu: float, sd: float, lo: int, hi: int) -> Dict[int, float]:
    """Normal(mu, sd) rounded to integers on [lo, hi] (continuity corrected, renormalised)."""
    d = {k: norm_cdf(k + 0.5, mu, sd) - norm_cdf(k - 0.5, mu, sd) for k in range(lo, hi + 1)}
    s = sum(d.values())
    return {k: v / s for k, v in d.items() if v > 1e-12}


# ---------------------------------------------------------------- tennis
def p_hold(p: float) -> float:
    """P(server wins a game) given P(server wins a point) = p."""
    q = 1 - p
    deuce = p * p / (p * p + q * q)
    return p**4 * (1 + 4 * q + 10 * q * q) + 20 * p**3 * q**3 * deuce


def p_tiebreak(pa: float, pb: float) -> float:
    """P(A wins a 7-point tiebreak). A serves first; serve alternates 1,2,2,2..."""
    @lru_cache(maxsize=None)
    def f(a: int, b: int) -> float:
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0
        if a >= 6 and b >= 6 and a == b:  # deuce-like state: two points, one each serve
            win2 = pa * (1 - pb)
            lose2 = (1 - pa) * pb
            return win2 / (win2 + lose2) if win2 + lose2 > 0 else 0.5
        n = a + b
        a_serves = (n % 4) in (0, 3)
        p = pa if a_serves else 1 - pb
        return p * f(a + 1, b) + (1 - p) * f(a, b + 1)
    return f(0, 0)


def set_distribution(pa: float, pb: float, a_serves_first: bool = True) -> Dict[Tuple[int, int], float]:
    """Distribution over final set scores (games A, games B), tiebreak at 6-6."""
    ha, hb = p_hold(pa), p_hold(pb)
    tb = p_tiebreak(pa, pb) if a_serves_first else 1 - p_tiebreak(pb, pa)
    out: Dict[Tuple[int, int], float] = {}
    states = {(0, 0): 1.0}
    while states:
        nxt: Dict[Tuple[int, int], float] = {}
        for (a, b), pr in states.items():
            if (a, b) == (6, 6):
                out[(7, 6)] = out.get((7, 6), 0) + pr * tb
                out[(6, 7)] = out.get((6, 7), 0) + pr * (1 - tb)
                continue
            a_serving = ((a + b) % 2 == 0) == a_serves_first
            pw = ha if a_serving else 1 - hb
            for (na, nb), p in (((a + 1, b), pw), ((a, b + 1), 1 - pw)):
                done = (na >= 6 and na - nb >= 2) or (nb >= 6 and nb - na >= 2) or na == 7 or nb == 7
                tgt = out if done else nxt
                tgt[(na, nb)] = tgt.get((na, nb), 0) + pr * p
        states = nxt
    return out


def match_distribution(pa: float, pb: float, best_of: int = 3) -> Dict[Tuple[int, int, int, int], float]:
    """Exact joint distribution over (sets A, sets B, games A, games B) at match end.

    Serve order is approximated as A serving first in every set (effect on prices is tiny).
    """
    return dict(_match_distribution(round(pa, 4), round(pb, 4), best_of))


@lru_cache(maxsize=2048)
def _match_distribution(pa: float, pb: float, best_of: int):
    sd = set_distribution(pa, pb)
    need = best_of // 2 + 1
    states = {(0, 0, 0, 0): 1.0}
    final: Dict[Tuple[int, int, int, int], float] = {}
    while states:
        nxt: Dict[Tuple[int, int, int, int], float] = {}
        for (sa, sb, ga, gb), pr in states.items():
            for (x, y), p in sd.items():
                key = (sa + (x > y), sb + (y > x), ga + x, gb + y)
                tgt = final if max(key[0], key[1]) == need else nxt
                tgt[key] = tgt.get(key, 0) + pr * p
        states = nxt
    return tuple(final.items())


@lru_cache(maxsize=64)
def _win_table(best_of: int, base: float):
    ds = [i / 400 for i in range(-100, 101)]          # serve-prob offsets -0.25..0.25
    return ds, [_quick_match_win(base + d, base - d, best_of) for d in ds]


def serve_probs_for_win_prob(target: float, best_of: int = 3, base: float = 0.63) -> Tuple[float, float]:
    """Symmetric serve-point probabilities (base+d, base-d) giving P(A wins match) = target."""
    ds, ws = _win_table(best_of, round(base * 200) / 200)   # 0.005 grid keeps the cache small
    target = min(max(target, ws[0]), ws[-1])
    for i in range(1, len(ds)):
        if ws[i] >= target:
            t = (target - ws[i - 1]) / ((ws[i] - ws[i - 1]) or 1.0)
            d = ds[i - 1] + t * (ds[i] - ds[i - 1])
            return base + d, base - d
    return base + ds[-1], base - ds[-1]


@lru_cache(maxsize=4096)
def _quick_match_win_cached(pa: float, pb: float, best_of: int) -> float:
    sd = set_distribution(pa, pb)
    ps = sum(p for (a, b), p in sd.items() if a > b)
    need = best_of // 2 + 1
    # P(win `need` sets before opponent) with per-set prob ps
    return sum(math.comb(need - 1 + k, k) * ps**need * (1 - ps) ** k for k in range(need))


def _quick_match_win(pa: float, pb: float, best_of: int) -> float:
    return _quick_match_win_cached(round(pa, 5), round(pb, 5), best_of)
