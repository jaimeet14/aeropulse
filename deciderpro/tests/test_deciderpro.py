import unittest

from deciderpro import synthetic
from deciderpro.backtest import implied_probs, run_backtest
from deciderpro.model import RatingModel
from deciderpro.risk import RiskConfig, RiskManager, kelly_fraction
from deciderpro.sports import SPORTS, get_sport


class KellyTests(unittest.TestCase):
    def test_no_edge_no_bet(self):
        self.assertEqual(kelly_fraction(0.5, 1.9), 0.0)

    def test_known_value(self):
        # p=0.6 at evens: f* = (0.6*1 - 0.4)/1 = 0.2
        self.assertAlmostEqual(kelly_fraction(0.6, 2.0), 0.2)


class ModelTests(unittest.TestCase):
    def test_probs_sum_to_one(self):
        for cfg in SPORTS.values():
            m = RatingModel(cfg)
            self.assertAlmostEqual(sum(m.predict("a", "b")), 1.0)

    def test_winner_rating_rises(self):
        m = RatingModel(get_sport("tennis"))
        before = m.predict("a", "b")[0]
        for _ in range(20):
            m.update("a", "b", 0)
        self.assertGreater(m.predict("a", "b")[0], before)


class RiskTests(unittest.TestCase):
    def test_stake_capped(self):
        rm = RiskManager(RiskConfig(), 1000)
        s = rm.stake(0.9, 2.0, 0.85, 1000, "2024-01-01")
        self.assertLessEqual(s, 1000 * RiskConfig().max_stake_frac + 1e-9)

    def test_kill_switch(self):
        rm = RiskManager(RiskConfig(max_drawdown=0.2), 1000)
        rm.observe(790)
        self.assertTrue(rm.halted)
        self.assertEqual(rm.stake(0.9, 2.0, 0.85, 790, "d"), 0.0)

    def test_daily_limit(self):
        rm = RiskManager(RiskConfig(max_daily_frac=0.03), 1000)
        total = sum(rm.stake(0.6, 2.0, 0.5, 1000, "d") for _ in range(10))
        self.assertLessEqual(total, 30 + 1e-9)


class BacktestTests(unittest.TestCase):
    def test_implied_probs_remove_margin(self):
        self.assertAlmostEqual(sum(implied_probs([1.9, 1.9])), 1.0)

    def test_perfect_book_has_no_edge(self):
        # a bookmaker with no error + margin: the model must not make money
        cfg = get_sport("football")
        ms = synthetic.generate(cfg, 3000, book_noise=0.0, seed=3)
        rep = run_backtest(ms, cfg)
        self.assertLessEqual(rep.end_bankroll, rep.start_bankroll * 1.05)

    def test_all_sports_run_and_never_nan(self):
        for name, cfg in SPORTS.items():
            ms = synthetic.generate(cfg, 800, seed=1)
            rep = run_backtest(ms, cfg)
            self.assertEqual(rep.matches, 800, name)
            self.assertGreater(rep.end_bankroll, 0, name)

    def test_model_learns_beats_naive(self):
        # after learning, log-loss should be well under the 3-way/2-way coin-flip baseline
        import math
        for name, base in (("football", math.log(3)), ("tennis", math.log(2))):
            cfg = get_sport(name)
            rep = run_backtest(synthetic.generate(cfg, 3000, seed=5), cfg)
            self.assertLess(rep.model_logloss, base, name)


if __name__ == "__main__":
    unittest.main()
