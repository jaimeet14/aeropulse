"""Live data connectors -> Event records (odds, injuries, lineups).

  OddsAPI       the-odds-api.com v4: moneyline/1x2, spreads, totals, btts, team totals and
                player props for football, basketball, cricket and tennis. Needs ODDS_API_KEY.
  ApiFootball   api-football (api-sports.io) v3: football injuries and confirmed lineups.
                Needs API_FOOTBALL_KEY.
  EspnInjuries  ESPN's public (unofficial, undocumented) injury feed: NBA and others. No key;
                the format can change without notice.

Cricket and tennis injuries have no reliable free feed; pass them in a JSON file instead
(see README). Team names differ between sources: use an aliases file to map them onto the
names your training data uses, or player/team history will not line up.
"""
import json
import os
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from .data import Event, Odds

UA = {"User-Agent": "DeciderPro/0.2"}


def http_json(url: str, headers: Dict = None, timeout: int = 20):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def apply_aliases(name: str, aliases: Dict[str, str]) -> str:
    return aliases.get(name, name)


# ============================================================================ The Odds API
ODDS_MARKETS = {
    "h2h": None, "spreads": "spread", "totals": "total", "btts": "btts", "team_totals": None,
    "player_points": "player_points", "player_rebounds": "player_rebounds",
    "player_assists": "player_assists", "player_threes": "player_threes",
    "player_goal_scorer_anytime": "player_goals", "player_shots_on_target": "player_shots_on_target",
    "batter_runs_scored": "player_runs", "player_aces": "player_aces",
}

SPORT_OF_KEY = {"soccer": "football", "basketball": "basketball", "cricket": "cricket", "tennis": "tennis"}


class OddsAPI:
    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str = None, regions: str = "uk,eu", fetch=http_json):
        self.key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.key:
            raise RuntimeError("set ODDS_API_KEY (https://the-odds-api.com)")
        self.regions, self.fetch = regions, fetch

    def events(self, sport_key: str, markets: str = "h2h,spreads,totals", props: str = "",
               bookmaker: str = "", aliases: Dict[str, str] = None) -> List[Event]:
        q = urllib.parse.urlencode({"apiKey": self.key, "regions": self.regions, "markets": markets,
                                    "oddsFormat": "decimal"})
        raw = self.fetch(f"{self.BASE}/sports/{sport_key}/odds?{q}")
        out = []
        for e in raw:
            if props:
                q2 = urllib.parse.urlencode({"apiKey": self.key, "regions": self.regions,
                                             "markets": props, "oddsFormat": "decimal"})
                extra = self.fetch(f"{self.BASE}/sports/{sport_key}/events/{e['id']}/odds?{q2}")
                e = {**e, "bookmakers": e.get("bookmakers", []) + extra.get("bookmakers", [])}
            out.append(parse_odds_event(e, sport_key, bookmaker, aliases or {}))
        return out


def parse_odds_event(e: Dict, sport_key: str, bookmaker: str = "", aliases: Dict[str, str] = None) -> Event:
    """One The Odds API event -> Event, keeping the best price per selection across bookmakers."""
    aliases = aliases or {}
    sport = SPORT_OF_KEY.get(sport_key.split("_")[0], sport_key)
    home, away = e["home_team"], e["away_team"]
    best: Dict[tuple, Odds] = {}
    for bk in e.get("bookmakers", []):
        if bookmaker and bk.get("key") != bookmaker:
            continue
        for mk in bk.get("markets", []):
            for o in _convert(mk["key"], mk.get("outcomes", []), home, away):
                k = (o.market, o.selection, o.line)
                if k not in best or o.price > best[k].price:
                    o.book = bk.get("key", "")
                    best[k] = o
    ev = Event(sport, e["commence_time"][:10], apply_aliases(home, aliases), apply_aliases(away, aliases),
               odds=list(best.values()), id=e.get("id", ""))
    ev.context["commence_time"] = e["commence_time"]
    return ev


def _convert(key: str, outcomes: List[Dict], home: str, away: str) -> List[Odds]:
    out = []
    side = {home: "home", away: "away"}
    if key == "h2h":
        mkt = "1x2" if any(o["name"] == "Draw" for o in outcomes) else "moneyline"
        for o in outcomes:
            sel = "draw" if o["name"] == "Draw" else side.get(o["name"])
            if sel:
                out.append(Odds(mkt, sel, float(o["price"])))
    elif key == "spreads":
        for o in outcomes:
            if o["name"] in side:
                out.append(Odds("spread", side[o["name"]], float(o["price"]), float(o["point"])))
    elif key == "totals":
        for o in outcomes:
            out.append(Odds("total", o["name"].lower(), float(o["price"]), float(o["point"])))
    elif key == "btts":
        for o in outcomes:
            out.append(Odds("btts", o["name"].lower(), float(o["price"])))
    elif key == "team_totals":
        for o in outcomes:
            s = side.get(o.get("description", ""))
            if s:
                out.append(Odds(f"team_total_{s}", o["name"].lower(), float(o["price"]), float(o["point"])))
    elif key in ODDS_MARKETS and ODDS_MARKETS[key]:
        mkt = ODDS_MARKETS[key]
        for o in outcomes:
            player = o.get("description")
            if not player:
                continue
            if key == "player_goal_scorer_anytime":
                if o["name"].lower() == "yes":
                    out.append(Odds(mkt, f"{player} over", float(o["price"]), 0.5))
            elif "point" in o:
                out.append(Odds(mkt, f"{player} {o['name'].lower()}", float(o["price"]), float(o["point"])))
    return out


# ============================================================================ API-Football
class ApiFootball:
    BASE = "https://v3.football.api-sports.io"

    def __init__(self, api_key: str = None, fetch=http_json):
        self.key = api_key or os.environ.get("API_FOOTBALL_KEY")
        if not self.key:
            raise RuntimeError("set API_FOOTBALL_KEY (https://www.api-football.com)")
        self.fetch = fetch

    def _get(self, path, **params):
        return self.fetch(f"{self.BASE}/{path}?{urllib.parse.urlencode(params)}",
                          {"x-apisports-key": self.key})

    def injuries(self, league: int, season: int, date: str, aliases=None) -> List[Dict]:
        return parse_api_football_injuries(self._get("injuries", league=league, season=season, date=date), aliases)

    def lineups(self, fixture: int, aliases=None) -> Dict[str, List[str]]:
        return parse_api_football_lineups(self._get("fixtures/lineups", fixture=fixture), aliases)


def parse_api_football_injuries(raw: Dict, aliases: Dict[str, str] = None) -> List[Dict]:
    out = []
    for r in raw.get("response", []):
        t = (r.get("player", {}).get("type") or "").lower()
        status = "out" if "missing" in t else "questionable" if "questionable" in t else "doubtful"
        out.append({"player": r["player"]["name"], "team": apply_aliases(r["team"]["name"], aliases or {}),
                    "status": status, "reason": r["player"].get("reason", "")})
    return out


def parse_api_football_lineups(raw: Dict, aliases: Dict[str, str] = None) -> Dict[str, List[str]]:
    """{team name: [starting XI + substitutes]} (map onto home/away when attaching)."""
    out = {}
    for t in raw.get("response", []):
        names = [p["player"]["name"] for p in t.get("startXI", []) + t.get("substitutes", [])]
        out[apply_aliases(t["team"]["name"], aliases or {})] = names
    return out


# ============================================================================ ESPN (unofficial)
ESPN_PATHS = {"basketball": "basketball/nba", "football": "soccer/eng.1"}


class EspnInjuries:
    def __init__(self, fetch=http_json):
        self.fetch = fetch

    def injuries(self, sport: str, path: Optional[str] = None, aliases=None) -> List[Dict]:
        p = path or ESPN_PATHS[sport]
        return parse_espn_injuries(self.fetch(f"https://site.api.espn.com/apis/site/v2/sports/{p}/injuries"), aliases)


def parse_espn_injuries(raw: Dict, aliases: Dict[str, str] = None) -> List[Dict]:
    out = []
    for team in raw.get("injuries", []):
        tname = apply_aliases(team.get("displayName", ""), aliases or {})
        for inj in team.get("injuries", []):
            out.append({"player": inj.get("athlete", {}).get("displayName", ""), "team": tname,
                        "status": (inj.get("status") or "").lower()})
    return out


# ============================================================================ attach
def attach_injuries(events: List[Event], injuries: List[Dict]) -> None:
    """Add each injury entry to the events its team plays in."""
    for ev in events:
        ev.injuries = [i for i in injuries if i.get("team") in (ev.home, ev.away)
                       or (ev.sport == "tennis" and i.get("player") in (ev.home, ev.away))]
