"""Per-sport configuration for the rating model."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SportConfig:
    name: str
    outcomes: int          # 3 = home/draw/away, 2 = home/away
    k: float               # Elo K-factor (how fast ratings move)
    home_adv: float        # Elo points of home advantage
    draw_base: float       # initial draw probability when teams are level (3-way only)
    scale: float = 400.0   # Elo logistic scale


SPORTS = {
    "football": SportConfig("football", 3, k=20.0, home_adv=65.0, draw_base=0.28),
    "basketball": SportConfig("basketball", 2, k=18.0, home_adv=90.0, draw_base=0.0),
    "cricket": SportConfig("cricket", 2, k=24.0, home_adv=40.0, draw_base=0.0),
    "tennis": SportConfig("tennis", 2, k=32.0, home_adv=0.0, draw_base=0.0),
}


def get_sport(name: str) -> SportConfig:
    try:
        return SPORTS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown sport {name!r}; choose from {sorted(SPORTS)}")
