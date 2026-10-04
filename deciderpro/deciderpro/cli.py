"""deciderpro  simulate | backtest | predict | fetch   (simulation / paper betting only)."""
import argparse
import json
import sys

from . import data, synthetic
from .backtest import run_backtest
from .decide import evaluate
from .engines import ENGINES, engine_from_dict, make_engine
from .risk import RiskConfig, RiskManager, TrustGate


def _risk(a) -> RiskConfig:
    return RiskConfig(kelly_mult=a.kelly, min_edge=a.min_edge, max_drawdown=a.max_drawdown,
                      min_games=a.min_games)


def _load_events(path: str, sport: str):
    if path.endswith(".csv"):
        if sport != "football":
            sys.exit("CSV import is football-data.co.uk only; use .jsonl for other sports")
        return data.load_football_data_csv(path)
    return [e for e in data.load_jsonl(path) if e.sport == sport]


def cmd_simulate(a):
    evs = synthetic.generate(a.sport, a.events, seed=a.seed, book_noise=a.book_noise,
                             book_injury_info=a.book_injury_info)
    data.save_jsonl(evs, a.out)
    print(f"wrote {len(evs)} synthetic {a.sport} events to {a.out} (SYNTHETIC: tests the machinery only)")


def cmd_backtest(a):
    evs = _load_events(a.data, a.sport)
    eng, gate = make_engine(a.sport), TrustGate(min_obs=a.gate_obs)
    rep = run_backtest(evs, eng, a.bankroll, _risk(a), gate)
    print(rep.summary())
    if a.save:
        with open(a.save, "w") as fh:
            json.dump({"engine": eng.to_dict(), "gate": gate.to_dict()}, fh)
        print(f"\nmodel state saved to {a.save} (use with: deciderpro predict --model {a.save})")


def cmd_predict(a):
    with open(a.model) as fh:
        st = json.load(fh)
    eng = engine_from_dict(st["engine"])
    gate = TrustGate()
    gate.load(st.get("gate", {}))
    evs = [e for e in data.load_jsonl(a.events) if e.sport == eng.sport]
    if a.injuries:
        from .providers import attach_injuries
        with open(a.injuries) as fh:
            attach_injuries(evs, json.load(fh))
    risk = _risk(a)
    rm = RiskManager(risk, a.bankroll)
    total = 0.0
    for i, ev in enumerate(evs):
        fc = eng.forecast(ev)
        print(f"\n=== {ev.date}  {ev.home} vs {ev.away}")
        if ev.injuries:
            print("  injuries: " + ", ".join(f"{x['player']} ({x.get('status')})" for x in ev.injuries))
        for mk, sel in fc.board().items():
            print(f"  {mk:<16} " + "  ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                                             for k, v in sel.items()))
        if not eng.confident(ev, risk.min_games):
            print("  (not enough history on one side: no bets)")
            continue
        picks = evaluate(fc, ev, risk)
        shown = 0
        for pk in picks:
            o = pk.odds
            line = f" {o.line:+g}" if o.line is not None and o.market in ("spread", "set_handicap") else \
                f" {o.line:g}" if o.line is not None else ""
            why = pk.reason or ("" if gate.allows(o.market) else "trust gate: model not beating market here yet")
            stake = 0.0 if why else rm.stake(pk.p_win, pk.p_push, o.price, a.bankroll, ev.date, ev.id or str(i))
            if stake > 0:
                total += stake
                print(f"  BET  {o.market:<16}{o.selection}{line} @ {o.price:.2f}  model {pk.p_win:.3f} "
                      f"vs market {pk.market_p:.3f}  edge {pk.edge:+.1%}  stake {stake:.2f}")
            elif a.all or (pk.edge >= risk.min_edge and shown < 5):
                shown += 1
                print(f"  --   {o.market:<16}{o.selection}{line} @ {o.price:.2f}  model {pk.p_win:.3f} "
                      f"edge {pk.edge:+.1%}  ({why or 'risk limits'})")
    print(f"\ntotal suggested stakes: {total:.2f} of bankroll {a.bankroll:.2f}  (paper betting only)")


def cmd_fetch(a):
    from . import providers as P
    aliases = json.load(open(a.aliases)) if a.aliases else {}
    evs = P.OddsAPI(regions=a.regions).events(a.odds_sport, a.markets, a.props, a.bookmaker, aliases)
    inj = []
    if a.injuries == "espn":
        inj = P.EspnInjuries().injuries(evs[0].sport if evs else "basketball", a.espn_path, aliases)
    elif a.injuries == "api-football":
        inj = P.ApiFootball().injuries(a.league, a.season, a.date, aliases)
    elif a.injuries:
        inj = json.load(open(a.injuries))
    P.attach_injuries(evs, inj)
    data.save_jsonl(evs, a.out)
    print(f"wrote {len(evs)} events with {sum(len(e.odds) for e in evs)} prices and "
          f"{len(inj)} injury entries to {a.out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="deciderpro", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def risk_args(p):
        p.add_argument("--bankroll", type=float, default=1000.0)
        p.add_argument("--kelly", type=float, default=0.25, help="Kelly multiplier (0.25 = quarter Kelly)")
        p.add_argument("--min-edge", type=float, default=0.03)
        p.add_argument("--max-drawdown", type=float, default=0.30)
        p.add_argument("--min-games", type=int, default=8)

    p = sub.add_parser("simulate", help="write synthetic events (players, injuries, all markets) to .jsonl")
    p.add_argument("--sport", choices=sorted(ENGINES), required=True)
    p.add_argument("--events", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--book-noise", type=float, default=1.0, help="0 = sharp bookmaker")
    p.add_argument("--book-injury-info", type=float, default=0.4, help="share of injury news the book prices in")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("backtest", help="walk-forward learn + bet over history; optionally save the model")
    p.add_argument("--sport", choices=sorted(ENGINES), required=True)
    p.add_argument("--data", required=True, help=".jsonl events or football-data.co.uk .csv")
    p.add_argument("--save", help="write learned model state (JSON)")
    p.add_argument("--gate-obs", type=int, default=200, help="observations before a market can be bet")
    risk_args(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("predict", help="price upcoming events and list value bets with stakes")
    p.add_argument("--model", required=True)
    p.add_argument("--events", required=True, help=".jsonl of upcoming events with odds")
    p.add_argument("--injuries", help="JSON list of {player, team, status} to attach")
    p.add_argument("--all", action="store_true", help="show every priced selection")
    risk_args(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("fetch", help="download live odds (+ injuries) into .jsonl (needs API keys)")
    p.add_argument("--odds-sport", required=True, help="The Odds API key, e.g. soccer_epl, basketball_nba, "
                                                        "cricket_ipl, tennis_atp_us_open")
    p.add_argument("--markets", default="h2h,spreads,totals")
    p.add_argument("--props", default="", help="e.g. player_points,player_rebounds or player_goal_scorer_anytime")
    p.add_argument("--regions", default="uk,eu")
    p.add_argument("--bookmaker", default="", help="only this bookmaker's prices (default: best price)")
    p.add_argument("--injuries", help="'espn', 'api-football' or a JSON file")
    p.add_argument("--espn-path", help="ESPN sport path, e.g. basketball/nba")
    p.add_argument("--league", type=int, help="api-football league id (39 = Premier League)")
    p.add_argument("--season", type=int)
    p.add_argument("--date", help="api-football date YYYY-MM-DD")
    p.add_argument("--aliases", help="JSON {source name: training name}")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fetch)

    a = ap.parse_args(argv)
    a.func(a)
    return 0
