# DeciderPro

Self-learning sports betting **simulator and paper-betting assistant** for football, basketball,
cricket and tennis. It prices every major market, factors in injuries, lineups, player stats and
match context, learns from each result, and only recommends bets in markets where it has proved
it out-predicts the bookmaker.

> **Simulation / paper betting only.** DeciderPro never places bets. Most models lose once the
> bookmaker margin is paid, and bookmakers limit accounts that win. Run it on real historical
> odds, then paper bet for several hundred bets before risking any money. Check that betting is
> legal where you live.

## What it models

| | Football | Basketball | Cricket (T20/ODI) | Tennis |
|---|---|---|---|---|
| Team/player strength | attack/defence (Poisson goals, Dixon-Coles) | offence/defence points | batting/bowling runs per format | overall + surface Elo, serve dominance |
| Context factors (learned) | rest, travel, rain, wind | rest, back-to-backs, travel, altitude | toss, dew, rain, venue scoring | surface, best-of-3/5, fatigue, injury |
| Injuries & lineups | ✔ learned absence impact per player | ✔ | ✔ | ✔ injury flag |
| Player props | goals (anytime scorer), shots… | points, rebounds, assists, threes | batter runs, wickets | aces, double faults |

**Markets:** 1X2, moneyline, double chance, draw-no-bet, spread / Asian handicap (half and whole
lines, with pushes), totals, team totals, both teams to score, correct score, tennis set betting,
set handicap, game handicap, total games, and player over/under props.

## How it decides

```
history ─► engine learns ratings, factor weights, player impact & stat rates
                 │
upcoming event ─►│ forecast: full distribution of the result (score / margin / total / sets / player stats)
                 ▼
     every offered price ─► model probability vs bookmaker probability (margin removed)
                 ▼
     filters: edge ≥ 3%, model within 12 pts of market, player confirmed (props), odds ≤ 8
                 ▼
     TRUST GATE: market type allowed only while the model has beaten the bookmaker's
                 accuracy there (Brier score, last 400 lines, statistically significant)
                 ▼
     stake: quarter-Kelly (push-aware), ≤2% per bet, ≤3% per event, ≤10% per day,
            kill switch at 30% drawdown
```

The **trust gate** is the self-learning safeguard. The system checks itself after every result and
stops betting markets where it is no better than the bookmaker. Against a sharp synthetic bookmaker
it places almost no bets, and a test enforces that.

**How the player layer works:**
- **Availability.** Injury-report statuses map to a chance of playing (out 0%, doubtful 25%,
  questionable 50%, probable 85%). A confirmed lineup overrides the injury report.
- **Absence impact.** The system learns how much each player is worth by comparing how the team
  performed against the model when that player was missing.
- **Teammate boost.** When a regular is out, the remaining players' prop projections go up.

## Run

Python 3.10+, no dependencies.

```bash
cd deciderpro

# 1. try the whole pipeline on synthetic data (players, injuries, every market)
python3 -m deciderpro simulate --sport basketball --events 2000 --out nba_synth.jsonl
python3 -m deciderpro backtest --sport basketball --data nba_synth.jsonl --save nba_model.json

# 2. real history: football-data.co.uk CSV (1X2, O/U 2.5, Asian handicap) or your own .jsonl
python3 -m deciderpro backtest --sport football --data E0.csv --save epl_model.json

# 3. live: fetch today's odds + injuries, then price them
export ODDS_API_KEY=...            # https://the-odds-api.com
python3 -m deciderpro fetch --odds-sport basketball_nba --props player_points,player_rebounds \
        --injuries espn --out today.jsonl
python3 -m deciderpro predict --model nba_model.json --events today.jsonl --bankroll 1000

python3 -m unittest discover -s tests
```

`predict` prints, for each game:
- fair probabilities for the headline markets
- every bet with its stake
- the best rejected candidates, with the reason each was rejected (trust gate, risk limit, player
  not confirmed, and so on)

### Live data sources (`deciderpro/providers.py`)

| Data | Source | Key |
|---|---|---|
| Odds, all sports (h2h, spreads, totals, btts, team totals, player props) | The Odds API v4 | `ODDS_API_KEY` |
| Football injuries and confirmed lineups | API-Football v3 (`--injuries api-football --league 39 --season 2026 --date …`) | `API_FOOTBALL_KEY` |
| NBA (and other) injuries | ESPN public feed (`--injuries espn`) | none, unofficial |
| Cricket/tennis injuries, anything else | JSON file `[{"player","team","status"}]` (`--injuries file.json`) | — |

Team names differ between sources. Pass `--aliases aliases.json` (`{"Manchester United": "Man United"}`)
so that live names match the names in your training history.

> The connectors are tested against sample responses in the providers' documented formats. They were
> not run against the live APIs while being built, so check the first real fetch.

### Event format (`.jsonl`, one event per line)

```json
{"sport": "basketball", "date": "2026-01-14", "home": "BOS", "away": "NYK",
 "context": {"rest_home": 1, "rest_away": 2, "venue": "TD Garden"},
 "injuries": [{"player": "J. Tatum", "team": "BOS", "status": "questionable"}],
 "lineups": {"home": ["..."], "away": ["..."]},
 "odds": [{"market": "spread", "selection": "home", "line": -4.5, "price": 1.91, "closing": 1.87},
          {"market": "player_points", "selection": "J. Brown over", "line": 24.5, "price": 1.87}],
 "result": {"home": 112, "away": 104, "lineups": {"home": ["..."], "away": ["..."]},
            "player_stats": {"J. Brown": {"points": 27, "rebounds": 6}}}}
```

Context keys: `rest_home/away` (days), `travel_km_home/away`, `altitude_km`, `neutral`, `rain`, `wind_ms`,
`venue`, `format` (t20/odi), `toss` (home/away), `dew`, `surface` (hard/clay/grass/indoor),
`best_of`, `sets_last_3d_home/away`. Tennis: `home`/`away` are player A/B, and `result.sets` holds games per set.

## Reading the backtest

- **Winner log-loss, model vs market.** If the model's number is higher, it is a worse
  forecaster than the bookmaker.
- **Closing-line value.** Positive CLV over hundreds of bets is the best evidence of a real edge.
- **Trust gate table.** Shows which markets the model has earned the right to bet.
- **Per-market table.** Bets, stake, profit and yield for each market.

Synthetic results (2,000 events per sport, seed 2) only show that the machinery behaves:
- The model learns.
- The trust gate blocks markets where the model is worse. In tennis it never bets.
- Drawdowns stay under the limit.

**They say nothing about real profit.**

## Not modelled yet
- Test-match cricket (draws, two innings)
- In-play betting
- Quarter-line Asian handicaps
- Correlation between bets on the same game, beyond the per-event stake cap
