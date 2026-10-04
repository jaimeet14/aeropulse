# DeciderPro

Self-learning value-bet **simulator** for football, basketball, cricket and tennis.

> **Simulation / paper betting only.** DeciderPro does not place real bets. Most betting models lose
> money once the bookmaker margin is paid, and bookmakers limit accounts that win. Before risking real
> money, run it on real historical odds and paper bet for several hundred bets. Check that betting
> is legal where you live.

## How it works

```
odds + results ─► rating model ─► compare to market odds ─► bet only if edge ≥ threshold
     ▲                                                        │
     └──────── learn from the result (after settling) ◄── fractional-Kelly stake + risk limits
```

| Module | Role |
|---|---|
| `model.py` | Elo ratings per team or player, plus an online-learned calibration (confidence temperature and draw rate). Updated after every match. |
| `risk.py` | Quarter-Kelly sizing, 2% cap per bet, 6% daily cap, odds cap, minimum history per team, a "distrust the model when it disagrees wildly with the market" rule, and a **30% drawdown kill switch**. |
| `backtest.py` | Walk-forward in date order, so there is no look-ahead. Reports bankroll, yield, drawdown, model vs. market log-loss, and closing-line value (CLV). |
| `synthetic.py` | Generated leagues for testing the machinery. The numbers it produces **mean nothing about real profit**. |
| `sports.py` | Per-sport settings (3-way football with draws; 2-way basketball, cricket and tennis). |

## Run

No dependencies, Python 3.10+.

```bash
cd deciderpro
python3 -m deciderpro --sport football --synthetic             # pipeline check
python3 -m deciderpro --sport football --csv E0.csv           # real data (football-data.co.uk)
python3 -m deciderpro --sport tennis --csv matches.csv        # generic 2-way format
python3 -m unittest discover -s tests
```

Options: `--bankroll`, `--kelly 0.25`, `--min-edge 0.03`, `--max-drawdown 0.30`, `--book-noise` (synthetic only).

### Data formats
- **Football:** [football-data.co.uk](https://www.football-data.co.uk/) CSVs as-is (`Date, HomeTeam, AwayTeam, FTR, B365H/D/A`, plus optional closing odds `B365CH/CD/CA`).
- **Basketball, cricket, tennis:** `date,home,away,winner,odds_home,odds_away[,close_home,close_away]`, where `winner` is `home` or `away` and dates are ISO `YYYY-MM-DD`.

## Reading the report
- **Model log-loss vs. market log-loss:** if the model isn't close to the market, it has no real edge.
- **Avg closing-line value:** consistently positive CLV over hundreds of bets is the best evidence of an edge. A profit with negative CLV is usually luck.
- **Kill switch tripped:** the bankroll fell 30% from its peak and betting stopped. Treat this as a signal to stop and investigate, not to raise stakes.

## Next steps
1. Plug in real historical odds for each sport and check log-loss and CLV against the market.
2. Add features beyond Elo: rest days, injuries, surface (tennis), venue and toss (cricket).
3. Add a paper-betting mode that logs live odds and outcomes daily.
