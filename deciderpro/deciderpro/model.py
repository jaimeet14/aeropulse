"""Self-learning probability model: Elo ratings + online-calibrated outcome probabilities.

Learning happens in two places, both updated after every settled match:
  * team ratings (Elo)
  * a calibration layer (temperature on the rating gap, and draw rate for 3-way sports)
    trained by online gradient descent on log-loss, so the model corrects its own over/under-confidence.
"""
import math
from collections import defaultdict
from typing import List

from .sports import SportConfig


class RatingModel:
    def __init__(self, cfg: SportConfig, lr: float = 0.01):
        self.cfg = cfg
        self.lr = lr
        self.ratings = defaultdict(lambda: 1500.0)
        self.temp = 1.0                       # calibration: scales the rating gap
        self.draw = cfg.draw_base             # learned draw rate at parity (3-way)
        self.seen = defaultdict(int)

    # --- prediction -------------------------------------------------------
    def _gap(self, home: str, away: str) -> float:
        return self.ratings[home] + self.cfg.home_adv - self.ratings[away]

    def _p_home_2way(self, gap: float) -> float:
        z = max(-30.0, min(30.0, self.temp * gap / self.cfg.scale * math.log(10)))
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, home: str, away: str) -> List[float]:
        gap = self._gap(home, away)
        p = self._p_home_2way(gap)
        if self.cfg.outcomes == 2:
            return [p, 1.0 - p]
        # 3-way: draws are likeliest between evenly matched sides
        d = self.draw * (1.0 - abs(2.0 * p - 1.0))
        d = max(0.02, min(0.45, d))
        return [p * (1.0 - d), d, (1.0 - p) * (1.0 - d)]

    def confidence(self, home: str, away: str, min_games: int) -> bool:
        """Do not trust ratings until both sides have enough history."""
        return self.seen[home] >= min_games and self.seen[away] >= min_games

    # --- learning ---------------------------------------------------------
    def update(self, home: str, away: str, result: int) -> None:
        gap = self._gap(home, away)
        p = self._p_home_2way(gap)
        k = self.cfg.k
        if self.cfg.outcomes == 3:
            score = (1.0, 0.5, 0.0)[result]
            # calibration: learn the draw rate
            probs = self.predict(home, away)
            parity = 1.0 - abs(2.0 * p - 1.0)
            if parity > 1e-6:
                target = 1.0 if result == 1 else 0.0
                self.draw += self.lr * (target - probs[1]) * parity
                self.draw = max(0.05, min(0.45, self.draw))
            exp_score = p  # expected score on the Elo scale
        else:
            score = 1.0 if result == 0 else 0.0
            exp_score = p
        # calibration: gradient step on log-loss of the home-vs-rest binary event
        hw = 1.0 if result == 0 else 0.0
        if self.cfg.outcomes == 2 or result != 1:
            grad = (hw - p) * gap / self.cfg.scale * math.log(10)
            self.temp = max(0.3, min(2.0, self.temp + self.lr * grad))
        # ratings
        delta = k * (score - exp_score)
        self.ratings[home] += delta
        self.ratings[away] -= delta
        self.seen[home] += 1
        self.seen[away] += 1
