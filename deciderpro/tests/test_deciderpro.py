import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from deciderpro import synthetic
from deciderpro.backtest import run_backtest
from deciderpro.cli import main
from deciderpro.data import Event, Odds
from deciderpro.decide import evaluate, market_probs
from deciderpro.dists import match_distribution, p_hold, p_tiebreak, score_matrix
from deciderpro.engines import ENGINES, engine_from_dict, make_engine
from deciderpro.markets import grade, settle
from deciderpro.providers import (parse_api_football_injuries, parse_api_football_lineups,
                                  parse_espn_injuries, parse_odds_event)
from deciderpro.risk import RiskConfig, RiskManager, TrustGate, kelly_fraction


class DistTests(unittest.TestCase):
    def test_score_matrix_normalised(self):
        self.assertAlmostEqual(sum(map(sum, score_matrix(1.6, 0.9))), 1.0)

    def test_tennis_known_values(self):
        self.assertAlmostEqual(p_hold(0.5), 0.5)
        self.assertAlmostEqual(p_hold(0.63), 0.7947, places=3)   # standard published value
        self.assertAlmostEqual(p_tiebreak(0.6, 0.6), 0.5, places=6)

    def test_better_server_wins_more_and_bo5_amplifies(self):
        d3 = match_distribution(0.66, 0.62, 3)
        d5 = match_distribution(0.66, 0.62, 5)
        w3 = sum(p for k, p in d3.items() if k[0] > k[1])
        w5 = sum(p for k, p in d5.items() if k[0] > k[1])
        self.assertGreater(w3, 0.5)
        self.assertGreater(w5, w3)
        self.assertAlmostEqual(sum(d5.values()), 1.0)


class MarketTests(unittest.TestCase):
    def test_grades(self):
        self.assertEqual(grade("spread", "home", -1.5, 2), "win")
        self.assertEqual(grade("spread", "home", -1.0, 1), "push")      # whole-line Asian handicap
        self.assertEqual(grade("spread", "away", 1.5, -1), "win")
        self.assertEqual(grade("dnb", "home", None, 0), "push")
        self.assertEqual(grade("double_chance", "X2", None, 0), "win")
        self.assertEqual(grade("1x2", "draw", None, 0), "win")
        self.assertEqual(grade("total", "under", 2.5, 3), "lose")
        self.assertEqual(grade("player_points", "A. B over", 20.5, 21), "win")

    def test_settle_football_and_tennis(self):
        r = {"home": 2, "away": 1}
        self.assertEqual(settle("football", "btts", "yes", None, r), "win")
        self.assertEqual(settle("football", "correct_score", "2-1", None, r), "win")
        t = {"home": 2, "away": 1, "sets": [[6, 4], [3, 6], [7, 6]]}
        self.assertEqual(settle("tennis", "total", "over", 31.5, t), "win")      # 32 games
        self.assertEqual(settle("tennis", "spread", "home", -2.5, t), "lose")   # won by 2 games
        self.assertEqual(settle("tennis", "set_betting", "2-1", None, t), "win")
        self.assertEqual(settle("tennis", "set_handicap", "home", -1.5, t), "lose")

    def test_props_void_when_player_absent(self):
        r = {"home": 100, "away": 90, "player_stats": {"X": {"points": 30}}}
        self.assertEqual(settle("basketball", "player_points", "Y over", 20.5, r), "void")
        self.assertEqual(settle("basketball", "player_points", "X over", 20.5, r), "win")

    def test_forecast_consistency(self):
        for sport in ENGINES:
            fc = make_engine(sport).forecast(Event(sport, "2026-01-01", "A", "B"))
            ph, _, _ = fc.prob("spread", "home", -0.5) if sport != "football" else fc.prob("dnb", "home")[:1] + (0, 0)
            for q, d in fc.m.items():
                self.assertAlmostEqual(sum(d.values()), 1.0, places=6, msg=f"{sport}/{q}")
            over, under = fc.prob("total", "over", 2.5 if sport == "football" else 200.5)[0], \
                fc.prob("total", "under", 2.5 if sport == "football" else 200.5)[0]
            self.assertAlmostEqual(over + under, 1.0, places=6)


class RiskTests(unittest.TestCase):
    def test_kelly(self):
        self.assertAlmostEqual(kelly_fraction(0.6, 2.0), 0.2)
        self.assertEqual(kelly_fraction(0.5, 1.9), 0.0)
        # a push chance shrinks risk but keeps the edge positive
        self.assertGreater(kelly_fraction(0.5, 2.2, push=0.2), 0)

    def test_caps_and_kill_switch(self):
        rm = RiskManager(RiskConfig(max_event_frac=0.03, max_bets_per_event=2), 1000)
        s = [rm.stake(0.7, 0, 2.0, 1000, "d", "e1") for _ in range(3)]
        self.assertLessEqual(s[0], 20 + 1e-9)
        self.assertLessEqual(sum(s), 30 + 1e-9)
        self.assertEqual(s[2], 0.0)
        rm.observe(690)
        self.assertTrue(rm.halted)
        self.assertEqual(rm.stake(0.9, 0, 2.0, 690, "d2", "e2"), 0.0)

    def test_trust_gate(self):
        g = TrustGate(window=100, min_obs=50)
        for i in range(60):
            won = i % 2 == 0
            g.record("total", 0.9 if won else 0.1, 0.5, won)   # model sharper than market
            g.record("spread", 0.1 if won else 0.9, 0.5, won)  # model worse than market
        self.assertTrue(g.allows("total"))
        self.assertFalse(g.allows("spread"))
        self.assertFalse(g.allows("btts"))  # never observed

    def test_devig(self):
        odds = [Odds("moneyline", "home", 1.9), Odds("moneyline", "away", 1.9)]
        mp = market_probs(odds)
        self.assertAlmostEqual(sum(mp.values()), 1.0)


class PlayerTests(unittest.TestCase):
    def test_injury_lowers_team_and_lineup_overrides(self):
        eng = make_engine("basketball")
        eng.players.apps["A|Star"] = 1.0
        eng.players.impact["A|Star"] = 6.0
        base = Event("basketball", "d", "A", "B")
        out = Event("basketball", "d", "A", "B", injuries=[{"player": "Star", "team": "A", "status": "out"}])
        q = Event("basketball", "d", "A", "B", injuries=[{"player": "Star", "team": "A", "status": "questionable"}])
        confirmed = Event("basketball", "d", "A", "B", injuries=out.injuries, lineups={"home": ["Star"]})
        w = lambda e: eng.forecast(e).m["winner"][1]
        self.assertLess(w(out), w(q))
        self.assertLess(w(q), w(base))
        self.assertAlmostEqual(w(confirmed), w(base))

    def test_impact_learned_from_absences(self):
        eng = make_engine("basketball")
        for i in range(60):
            star_plays = i % 2 == 0
            lineup = ["S", "x"] if star_plays else ["x"]
            ev = Event("basketball", f"2026-01-{i:03d}", "A", "B",
                       result={"home": 115 if star_plays else 103, "away": 108, "lineups": {"home": lineup, "away": ["y"]}})
            eng.update(ev)
        self.assertGreater(eng.players.impact.get("A|S", 0), 2.0)


class ProviderTests(unittest.TestCase):
    def test_odds_api_event(self):
        raw = {"id": "e1", "commence_time": "2026-05-01T19:00:00Z", "home_team": "Boston Celtics",
               "away_team": "New York Knicks", "bookmakers": [
                   {"key": "b1", "markets": [
                       {"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": 1.5},
                                                   {"name": "New York Knicks", "price": 2.7}]},
                       {"key": "spreads", "outcomes": [{"name": "Boston Celtics", "price": 1.91, "point": -5.5},
                                                       {"name": "New York Knicks", "price": 1.91, "point": 5.5}]},
                       {"key": "totals", "outcomes": [{"name": "Over", "price": 1.9, "point": 221.5},
                                                      {"name": "Under", "price": 1.9, "point": 221.5}]},
                       {"key": "player_points", "outcomes": [
                           {"name": "Over", "description": "Jayson Tatum", "price": 1.87, "point": 27.5}]}]},
                   {"key": "b2", "markets": [{"key": "h2h", "outcomes": [
                       {"name": "Boston Celtics", "price": 1.55}, {"name": "New York Knicks", "price": 2.6}]}]}]}
        ev = parse_odds_event(raw, "basketball_nba", aliases={"Boston Celtics": "BOS"})
        self.assertEqual(ev.home, "BOS")
        got = {(o.market, o.selection, o.line): o.price for o in ev.odds}
        self.assertEqual(got[("moneyline", "home", None)], 1.55)   # best price across books
        self.assertEqual(got[("spread", "away", 5.5)], 1.91)
        self.assertEqual(got[("player_points", "Jayson Tatum over", 27.5)], 1.87)

    def test_soccer_h2h_is_1x2(self):
        raw = {"id": "e", "commence_time": "2026-05-01T15:00:00Z", "home_team": "A", "away_team": "B",
               "bookmakers": [{"key": "b", "markets": [{"key": "h2h", "outcomes": [
                   {"name": "A", "price": 2.0}, {"name": "Draw", "price": 3.4}, {"name": "B", "price": 3.9}]}]}]}
        ev = parse_odds_event(raw, "soccer_epl")
        self.assertEqual(ev.sport, "football")
        self.assertEqual({o.market for o in ev.odds}, {"1x2"})

    def test_injury_feeds(self):
        af = {"response": [{"player": {"name": "B. Saka", "type": "Missing Fixture", "reason": "Hamstring"},
                            "team": {"name": "Arsenal"}}]}
        self.assertEqual(parse_api_football_injuries(af)[0]["status"], "out")
        lu = {"response": [{"team": {"name": "Arsenal"}, "startXI": [{"player": {"name": "D. Raya"}}],
                            "substitutes": [{"player": {"name": "K. Havertz"}}]}]}
        self.assertEqual(parse_api_football_lineups(lu)["Arsenal"], ["D. Raya", "K. Havertz"])
        espn = {"injuries": [{"displayName": "Boston Celtics", "injuries": [
            {"athlete": {"displayName": "Jayson Tatum"}, "status": "Day-To-Day"}]}]}
        self.assertEqual(parse_espn_injuries(espn), [{"player": "Jayson Tatum", "team": "Boston Celtics",
                                                       "status": "day-to-day"}])


class EndToEndTests(unittest.TestCase):
    def test_every_sport_backtests_and_round_trips(self):
        for sport in ENGINES:
            evs = synthetic.generate(sport, 300, seed=1)
            eng = make_engine(sport)
            rep = run_backtest(evs, eng, risk=RiskConfig(min_games=3), gate=TrustGate(min_obs=20))
            self.assertEqual(rep.events, 300, sport)
            self.assertGreater(rep.end_bankroll, 0, sport)
            clone = engine_from_dict(json.loads(json.dumps(eng.to_dict())))
            ev = evs[-1]
            self.assertAlmostEqual(clone.forecast(ev).m["winner"][1], eng.forecast(ev).m["winner"][1], places=9)

    def test_sharp_book_is_not_beaten(self):
        # a bookmaker with true ratings and full injury news: the system must not find a real edge
        evs = synthetic.generate("football", 1500, seed=3, book_noise=0.0, book_injury_info=1.0)
        rep = run_backtest(evs, make_engine("football"))
        self.assertLess(rep.bets, 60)
        self.assertLess(rep.end_bankroll, rep.start_bankroll * 1.05)

    def test_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            hist, model = os.path.join(d, "h.jsonl"), os.path.join(d, "m.json")
            with redirect_stdout(io.StringIO()):
                main(["simulate", "--sport", "cricket", "--events", "200", "--out", hist])
                main(["backtest", "--sport", "cricket", "--data", hist, "--save", model, "--gate-obs", "20"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["predict", "--model", model, "--events", hist, "--all"])
            self.assertIn("moneyline", buf.getvalue())
            self.assertIn("total suggested stakes", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
