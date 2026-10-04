"""Synthetic league generator for pipeline testing.

IMPORTANT: results here say nothing about real-world profitability. The generator has a hidden
'true strength' per team; the simulated bookmaker sees that strength through noise
(`book_noise`) and adds a margin. With book_noise=0 the book is perfect and the model should
lose roughly the margin; with a noisy book there is a learnable edge. Use it to check the
machinery (learning, Kelly sizing, kill switch), not to estimate returns.
"""
import math
import random
from datetime import date, timedelta
from typing import List

from .data import Match
from .sports import SportConfig


def _probs(cfg: SportConfig, gap: float, draw: float) -> List[float]:
    p = 1.0 / (1.0 + 10 ** (-gap / cfg.scale))
    if cfg.outcomes == 2:
        return [p, 1.0 - p]
    d = draw * (1.0 - abs(2 * p - 1.0))
    return [p * (1 - d), d, (1 - p) * (1 - d)]


def generate(cfg: SportConfig, n_matches: int = 3000, n_teams: int = 20, margin: float = 0.05,
             book_noise: float = 60.0, seed: int = 7) -> List[Match]:
    rng = random.Random(seed)
    strength = {f"T{i}": rng.gauss(1500, 120) for i in range(n_teams)}
    teams = list(strength)
    start = date(2020, 1, 1)
    out: List[Match] = []
    for i in range(n_matches):
        h, a = rng.sample(teams, 2)
        gap = strength[h] + cfg.home_adv - strength[a]
        true_p = _probs(cfg, gap, cfg.draw_base)
        # bookmaker's noisy view of the same gap
        view = _probs(cfg, gap + rng.gauss(0, book_noise), cfg.draw_base)
        close = [p * (1 + margin) for p in view]
        close = [1.0 / min(p, 0.97) for p in close]
        opening = [1.0 / min(p * (1 + margin) * math.exp(rng.gauss(0, 0.03)), 0.97) for p in view]
        r, acc, res = rng.random(), 0.0, len(true_p) - 1
        for idx, p in enumerate(true_p):
            acc += p
            if r < acc:
                res = idx
                break
        out.append(Match((start + timedelta(days=i // 4)).isoformat(), h, a, res, opening, close))
        # slow drift in team strength keeps the learning problem non-stationary
        for t in (h, a):
            strength[t] += rng.gauss(0, 2.0)
    return out
