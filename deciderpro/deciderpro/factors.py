"""Context factors (rest, back-to-backs, travel, toss, weather, injuries...) as an online linear model.

Each engine turns an event's context into a small feature vector. The weights start at
sensible priors and are learned by SGD on the residual the base ratings could not explain,
so the data decides how much a back-to-back or a won toss is really worth.
"""
from typing import Dict


class FactorModel:
    def __init__(self, priors: Dict[str, float], lr: float = 0.002, l2: float = 1e-4):
        self.w = dict(priors)
        self.lr, self.l2 = lr, l2

    def adjust(self, x: Dict[str, float]) -> float:
        return sum(self.w.get(k, 0.0) * v for k, v in x.items())

    def learn(self, x: Dict[str, float], residual: float) -> None:
        for k, v in x.items():
            if v and k in self.w:
                self.w[k] += self.lr * (residual * v - self.l2 * self.w[k])

    def to_dict(self) -> Dict:
        return dict(self.w)

    def load(self, d: Dict) -> None:
        self.w.update(d)


def num(ctx: Dict, key: str, default: float = 0.0) -> float:
    try:
        return float(ctx.get(key, default))
    except (TypeError, ValueError):
        return default


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
