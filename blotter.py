#!/usr/bin/env python3
"""
fantasydash - live cross-league fantasy exposure.

Run:  python blotter.py
Then: http://127.0.0.1:5000

Config lives in CONFIG below. Manual (non-API) teams go in manual.json.
ESPN leagues go in espn.json (see espn.example.json; needs espn_s2 + SWID).
"""

import base64
import json
import math
import os
import re
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np
import requests
from flask import Flask, Response, jsonify, request

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

CONFIG = {
    "sleeper_username": os.environ.get("SLEEPER_USERNAME", "Chrispl"),
    "season": "2026",
    "poll_seconds": 20,          # server-side cache TTL
    # League ids you want forced into guillotine/pool mode, if auto-detect fails.
    "force_pool_leagues": [],
    # League ids to ignore entirely (e.g. best ball, dormant dynasty).
    "exclude_leagues": [],
    "sims": 5000,                # Monte Carlo draws per league
    "log_seconds": 60,           # min gap between calibration snapshots (0 = off)
    # Guillotine leagues whose /l/<id> page is open (no password) so
    # leaguemates can pick their own team and see the chop picture.
    "shared_leagues": ["1400335104223485952"],   # Paris in 1795v2
    # How much you care, per league id (default 1.0). Scales the rooting
    # interest of every player in that league. Rebuilding dynasty teams are
    # "my guys, but not this year".
    "league_weights": {
        "1314037619989958656": 0.25,   # Justin Fields Appreciation League
        "1313334879160651776": 0.25,   # Dynasty Degenrates
    },
}

HERE = Path(__file__).parent
DATA = Path(os.environ.get("DATA_DIR", HERE))      # persistent disk on Fly (/data)
PLAYER_CACHE = DATA / "players_nfl.json"
MANUAL_FILE = HERE / "manual.json"
ESPN_FILE = HERE / "espn.json"
LOG_DB = DATA / "blotter.sqlite"
BASE = "https://api.sleeper.app/v1"
PROJ_BASE = "https://api.sleeper.com/projections/nfl"   # undocumented, same data the app uses

app = Flask(__name__)
_cache = {"ts": 0.0, "data": None}
_build_lock = threading.Lock()
_proj_cache = {"key": None, "ts": 0.0, "data": {}}
_games_cache = {"key": None, "ts": 0.0, "data": {}}
_stats_cache = {"key": None, "ts": 0.0, "data": {}}
_log_last = {"ts": 0.0}
OT_TEAMS = set()          # teams currently in overtime, for the "left" column
_plays_cache = {}         # espn game id -> {"ts", "plays", "teams"}
FEED_MAX = 60             # plays shown in the feed


# ----------------------------------------------------------------------------
# Sleeper API
# ----------------------------------------------------------------------------

def get(path, base=BASE, tries=3):
    for i in range(tries):
        try:
            r = requests.get(f"{base}{path}", timeout=10,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            if i == tries - 1:
                raise
            time.sleep(0.5)


def load_players():
    """The /players/nfl payload is ~5MB. Fetch once a day, cache to disk."""
    if PLAYER_CACHE.exists() and time.time() - PLAYER_CACHE.stat().st_mtime < 86400:
        return json.loads(PLAYER_CACHE.read_text())
    data = get("/players/nfl")
    slim = {
        pid: {
            "name": p.get("full_name") or p.get("last_name") or pid,
            "pos": p.get("position") or "",
            "team": p.get("team") or "FA",
            "espn_id": p.get("espn_id"),
        }
        for pid, p in data.items()
    }
    PLAYER_CACHE.write_text(json.dumps(slim))
    return slim


def load_projections(season, week):
    """
    Week projections for every player: {player_id: {stat: value}}. The stat
    keys use the same vocabulary as a league's scoring_settings, so a
    league-specific projection is just sum(stat * weight). Re-fetched every
    15 min since Sleeper revises them as inactives come in.
    """
    key = (season, week)
    if _proj_cache["key"] == key and time.time() - _proj_cache["ts"] < 900:
        return _proj_cache["data"]
    qs = "&".join(f"position[]={p}" for p in ("QB", "RB", "WR", "TE", "K", "DEF"))
    rows = get(f"/{season}/{week}?season_type=regular&{qs}", base=PROJ_BASE)
    data = {row["player_id"]: row.get("stats") or {} for row in rows}
    _proj_cache.update(key=key, ts=time.time(), data=data)
    return data


def load_stats(season, week):
    """Live actual stat lines for every player who has recorded one this week,
    with Sleeper's pts_ppr / pts_half_ppr / pts_std precomputed. Cached for
    one poll so the feed sees stat lines move with the points."""
    key = (season, week)
    if _stats_cache["key"] == key and time.time() - _stats_cache["ts"] < CONFIG["poll_seconds"]:
        return _stats_cache["data"]
    qs = "&".join(f"position[]={p}" for p in ("QB", "RB", "WR", "TE", "K", "DEF"))
    rows = get(f"/{season}/{week}?season_type=regular&{qs}", base="https://api.sleeper.com/stats/nfl")
    data = {row["player_id"]: row.get("stats") or {} for row in rows}
    _stats_cache.update(key=key, ts=time.time(), data=data)
    return data


def load_game_state(season, week):
    """
    {team: fraction of game remaining}. 1.0 before kickoff, 0.0 at final,
    quarter + clock in between. Teams on bye are absent (treated as 1.0,
    harmless since they have no projection). Cached 60s.
    """
    key = (season, week)
    if _games_cache["key"] == key and time.time() - _games_cache["ts"] < 60:
        return _games_cache["data"]
    out = {}
    for g in get(f"/scores/nfl/regular/{season}/{week}", base="https://api.sleeper.com"):
        md = g.get("metadata") or {}
        if g.get("status") == "complete" or md.get("is_over"):
            frac = 0.0
        elif g.get("status") == "pre_game" or not md.get("has_started"):
            frac = 1.0
        else:
            q = md.get("quarter_num") or 1
            try:
                mm, ss = (md.get("time_remaining") or "15:00").split(":")
                clock = int(mm) + int(ss) / 60
            except ValueError:
                clock = 15.0
            # Overtime: quarter_num 5, 10-minute period. Treat the OT clock as
            # remaining time against the regulation length so projections keep
            # a small tail rather than snapping to final.
            frac = clock / 60 if q > 4 else ((4 - q) * 15 + clock) / 60
            frac = max(0.0, min(1.0, frac))
        for t in (md.get("home_team"), md.get("away_team")):
            if t:
                out[t] = frac
                if md.get("is_overtime") and frac > 0:
                    OT_TEAMS.add(t)
                else:
                    OT_TEAMS.discard(t)
    _games_cache.update(key=key, ts=time.time(), data=out)
    return out


def proj_pts(pid, projections, scoring):
    stats = projections.get(pid)
    if not stats:
        return None
    return round(sum(v * scoring.get(k, 0) for k, v in stats.items() if k in scoring), 2)


def blended(pid, actual, projections, scoring, players, games):
    """
    Live projected final for one player: points so far plus the remaining
    share of the pregame projection. DEF ids are team codes, so they hit
    the games map directly.
    """
    proj = proj_pts(pid, projections, scoring) or 0.0
    team = pid if pid in games else players.get(pid, {}).get("team")
    rem = games.get(team, 1.0)
    return round((actual.get(pid) or 0.0) + proj * rem, 2)


# Slot -> eligible positions. Anything not listed is treated as its own position.
ELIGIBLE = {
    "FLEX": {"RB", "WR", "TE"}, "WRRB_FLEX": {"RB", "WR"}, "REC_FLEX": {"WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"}, "IDP_FLEX": {"DL", "LB", "DB"},
}
NON_SLOTS = {"BN", "IR", "TAXI"}


# Weekly scoring spread as a fraction of projection, by position. Rough
# empirical values: QBs are the steadiest, TEs the most boom/bust.
CV = {"QB": 0.45, "RB": 0.65, "WR": 0.75, "TE": 0.85, "K": 0.55, "DEF": 0.70}


def player_dist(d):
    """(mean, sd) of a starter's REMAINING points, or None if nothing left.
    Variance scales with time left, like a random walk."""
    mu = d["proj"] * d["rem"]
    if mu <= 0:
        return None
    sd = CV.get(d["pos"], 0.7) * d["proj"] * math.sqrt(d["rem"])
    return mu, sd


def simulate(teams, n):
    """
    teams: {rid: {pid: {"act","proj","rem","pos"}}} -> {rid: ndarray of n
    simulated finals}. Remaining points are gamma-distributed (right-skewed,
    never negative). Fixed seed so refresh-to-refresh moves reflect data,
    not sampling noise.
    """
    rng = np.random.default_rng(7)
    out = {}
    for rid, det in teams.items():
        tot = np.full(n, sum(d["act"] for d in det.values()))
        for d in det.values():
            dist = player_dist(d)
            if dist:
                mu, sd = dist
                tot += rng.gamma((mu / sd) ** 2, sd * sd / mu, n)
        out[rid] = tot
    return out


def to_play(det):
    """(starters with any game left, of which currently in progress)."""
    left = [d for d in det.values() if d["rem"] > 0 and d["proj"] > 0]
    return len(left), sum(1 for d in left if d["rem"] < 1)


def stat_line(st, pos):
    """Compact box-score line from Sleeper's live stats, e.g.
    '7 car 67 yd 1 TD, 3/5 rec 15 yd, 1 FL'."""
    if not st:
        return ""
    g = lambda k: int(st.get(k) or 0)
    td = lambda k: f" {g(k)} TD" if g(k) else ""
    parts = []
    if g("pass_att"):
        parts.append(f"{g('pass_cmp')}/{g('pass_att')} {g('pass_yd')} pyd{td('pass_td')}"
                     + (f" {g('pass_int')} INT" if g("pass_int") else ""))
    if g("rush_att"):
        parts.append(f"{g('rush_att')} car {g('rush_yd')} yd{td('rush_td')}")
    if g("rec_tgt") or g("rec"):
        parts.append(f"{g('rec')}/{g('rec_tgt')} rec {g('rec_yd')} yd{td('rec_td')}")
    if g("fum_lost"):
        parts.append(f"{g('fum_lost')} FL")
    if pos == "K":
        parts.append(f"FG {g('fgm')}/{g('fga')}, XP {g('xpm')}")
    if pos == "DEF":
        parts.append(f"{g('pts_allow')} PA, {g('sack')} sk, {g('int') + g('fum_rec')} TO{td('def_td')}")
    return ", ".join(parts)


def lineup_rows(det, players, stats):
    """Per-starter rows for the drill-down, in lineup order."""
    rows = []
    for pid, d in det.items():
        p = players.get(pid, {})
        rows.append({
            "pid": pid, "name": p.get("name", pid), "pos": p.get("pos", ""), "team": p.get("team", ""),
            "act": round(d["act"], 2), "proj": round(d["proj"], 2),
            "final": round(d["act"] + d["proj"] * d["rem"], 2), "rem": round(d["rem"], 3),
            "ot": (pid if pid in OT_TEAMS else p.get("team")) in OT_TEAMS,
            "line": stat_line(stats.get(pid), p.get("pos", "")),
        })
    return rows


def point_value(me, others, delta=2.0):
    """
    Percentage points of P(me beats `others`) gained per extra fantasy point
    for me, estimated by nudging my simulated total +-delta. `others` is an
    array to beat: the opponent in h2h, the field minimum in a pool.
    """
    up = np.mean(me + delta > others)
    dn = np.mean(me - delta > others)
    return round(100 * float(up - dn) / (2 * delta), 3)


def rival_point_value(me, rival, rest, delta=2.0):
    """Same, for a point scored by a specific rival in a pool: negative when
    that rival is near the chop line, ~0 when they are safe. `rest` is the
    min of everyone else."""
    up = np.mean(me > np.minimum(rest, rival + delta))
    dn = np.mean(me > np.minimum(rest, rival - delta))
    return round(100 * float(up - dn) / (2 * delta), 3)


ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_UA = {"User-Agent": "curl/8.0"}          # Akamai rejects fake browser UAs here
FEED_SKIP_TYPES = {"Kickoff", "Pass Incompletion", "Sack", "Penalty", "Punt", "Kneel", "Spike",
                   "Official Timeout", "Coin Toss"}
PLAY_ROLES = {"passer": "pass", "rusher": "rush", "receiver": "rec", "kicker": "kick",
              "returner": "ret", "scorer": "score", "fumbler": "fum", "puntReturner": "ret",
              "kickReturner": "ret", "interceptedBy": "int", "recoveredBy": "rec fum"}


def load_plays():
    """
    Recent plays from every in-progress game, via ESPN's public play-by-play.
    Each game's play list is cached one poll; finished games keep their last
    cached plays for the rest of the process so the feed doesn't empty at
    the final whistle.
    """
    try:
        sb = requests.get(f"{ESPN_SITE}/scoreboard", headers=ESPN_UA, timeout=10).json()
    except Exception:
        return _plays_cache
    for ev in sb.get("events", []):
        gid = ev["id"]
        state = ev["status"]["type"]["state"]
        c = _plays_cache.get(gid)
        if state != "in" and c:
            continue                      # keep what we had
        if state != "in":
            continue
        if c and time.time() - c["ts"] < CONFIG["poll_seconds"]:
            continue
        try:
            sm = requests.get(f"{ESPN_SITE}/summary?event={gid}", headers=ESPN_UA, timeout=15).json()
        except Exception:
            continue
        drives = sm.get("drives") or {}
        raw = []
        for d in drives.get("previous") or []:
            raw.extend(d.get("plays") or [])
        if drives.get("current"):
            raw.extend(drives["current"].get("plays") or [])
        seen, plays = set(), []
        for pl in raw:                    # current drive is also in previous
            if pl.get("id") in seen:
                continue
            seen.add(pl.get("id"))
            plays.append(pl)
        teams = {c["id"]: ESPN_TEAM_FIX.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
                 for c in (sm.get("header", {}).get("competitions") or [{}])[0].get("competitors", [])}
        _plays_cache[gid] = {"ts": time.time(), "plays": plays[-80:], "teams": teams,
                             "label": ev.get("shortName", "")}
    return _plays_cache


_INITIAL_LAST = re.compile(r"\b([A-Z])\.\s?([A-Z][A-Za-z'\-]+)")


def build_feed(book, players=None):
    """
    Plays involving anyone in `book` (dicts with id/name/team/for/against/
    pts), newest first. Players are matched
    from ESPN's participant list (full names), falling back to the
    'J.Gibbs' tokens in the play text matched by initial + surname + team.
    """
    by_name, by_init = {}, {}
    for p in book:
        by_name[norm(p["name"])] = p
        last = norm(p["name"].split()[-1]) if p["name"].split() else ""
        by_init[(p["name"][:1].upper(), last, p["team"])] = p
    out = []
    for gid, g in load_plays().items():
        game_teams = set(g["teams"].values())
        for pl in g["plays"]:
            text = (pl.get("text") or "").strip()
            ptype = (pl.get("type") or {}).get("text", "")
            if not text or ptype in ("Two-minute warning", "End Period", "End of Half", "End of Game", "Timeout"):
                continue
            # Only plays that move a fantasy score: drop incompletions, sacks,
            # penalties, punts, kneels, spikes and non-scoring kickoffs.
            if not pl.get("scoringPlay") and (
                ptype in FEED_SKIP_TYPES
                or "pass incomplete" in text.lower()
                or "penalty" in ptype.lower()
            ):
                continue
            hits, roles = {}, {}
            for part in pl.get("participants") or []:
                if part.get("type") in PLAY_ROLES:
                    hit = by_name.get(norm(part.get("athlete", {}).get("displayName", "")))
                    if hit:
                        hits[hit["id"]] = hit
                        roles.setdefault(hit["id"], PLAY_ROLES[part["type"]])
            if not hits:
                # No participant list: read 'J.Love pass ... to C.Watson' style text.
                # First name token is the passer/rusher, the one after 'to' the receiver.
                tokens = _INITIAL_LAST.findall(text)
                for i, (ini, last) in enumerate(tokens):
                    for t in game_teams:
                        hit = by_init.get((ini, norm(last), t))
                        if hit:
                            hits[hit["id"]] = hit
                            if i == 0:
                                roles.setdefault(hit["id"], "pass" if " pass " in text else "rush")
                            elif f"to {ini}.{last}" in text:
                                roles.setdefault(hit["id"], "rec")
            if not hits:
                continue
            try:
                ts = time.mktime(time.strptime(pl.get("wallclock", "")[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
                if time.localtime(ts).tm_isdst:
                    ts += 3600
            except (ValueError, TypeError):
                ts = time.time()
            q = pl.get("period", {}).get("number")
            out.append({
                "id": pl.get("id"), "ts": ts, "t": time.strftime("%H:%M", time.localtime(ts)),
                "game": g["label"], "q": f"Q{q}" if q and q <= 4 else "OT",
                "clock": (pl.get("clock") or {}).get("displayValue", ""),
                "text": re.sub(r"^\((?:Shotgun|No Huddle|No Huddle, Shotgun)\)\s*", "", text),
                "scoring": bool(pl.get("scoringPlay")),
                "players": [{"id": h["id"], "name": h["name"], "for": h["for"], "against": h["against"],
                             "pts": h["pts"], "role": roles.get(h["id"], "")} for h in hits.values()],
            })
    out.sort(key=lambda e: -e["ts"])
    return out[:FEED_MAX]


def win_pct(a, b):
    return round(100 * float(np.mean(a > b) + 0.5 * np.mean(a == b)), 1)


def chop_line(sorted_field, idx, key):
    """
    Distance to the chop line, signed. Positive: your cushion over the
    lowest team. Negative: you ARE the lowest, and this is how far behind
    the next team up you sit. Returns (margin, reference team).
    """
    if idx == 0 and len(sorted_field) > 1:
        ref = sorted_field[1]
        return round(sorted_field[0][key] - ref[key], 2), ref
    ref = sorted_field[0]
    return round(sorted_field[idx][key] - ref[key], 2), ref


def best_lineup(player_ids, slots, value, players):
    """
    Best-ball lineup: fill the most restrictive slots first, each with the
    highest-valued eligible player left. Exact for the usual nested
    QB/RB/WR/TE -> FLEX -> SUPER_FLEX ladder.
    """
    avail = {p: value(p) for p in player_ids if p and p != "0"}
    order = sorted((s for s in slots if s not in NON_SLOTS),
                   key=lambda s: len(ELIGIBLE.get(s, {s})))
    chosen = []
    for slot in order:
        elig = ELIGIBLE.get(slot, {slot})
        cands = [p for p in avail if players.get(p, {}).get("pos") in elig]
        if cands:
            best = max(cands, key=avail.get)
            chosen.append(best)
            del avail[best]
    return chosen


_SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv|v)\.?$", re.I)


def norm(name):
    """Loose key: strip accents, generational suffixes (ESPN keeps "Jr."/"III",
    Sleeper drops them), and everything but letters."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", _SUFFIX.sub("", s.strip()).lower())


# ----------------------------------------------------------------------------
# Manual teams (eliminator.gg, or anything else without an API)
# ----------------------------------------------------------------------------

# Custom scoring tables for manual teams, in Sleeper's stat vocabulary.
# Eliminator Open, published 2026-08-24. Yardage bonuses are computed from
# raw yards below since Sleeper only emits its bonus flags once triggered.
ELIMINATOR = {
    "pass_td": 4, "pass_yd": 0.04, "pass_int": -1,
    "rush_td": 6, "rush_yd": 0.1, "rec_td": 6, "rec_yd": 0.1, "rec": 1,
    "pr_td": 6, "kr_td": 6, "fum_lost": -1, "fum_rec_td": 6,
    "pass_2pt": 2, "rush_2pt": 2, "rec_2pt": 2,
}
ELIMINATOR_BONUS = (("pass_yd", 300, 3), ("rush_yd", 100, 3), ("rec_yd", 100, 3))


def eliminator_pts(stats):
    pts = sum(v * ELIMINATOR[k] for k, v in stats.items() if k in ELIMINATOR)
    pts += sum(b for k, thr, b in ELIMINATOR_BONUS if (stats.get(k) or 0) >= thr)
    return pts


def manual_score(stats, scoring):
    """scoring: ppr | half_ppr | std use Sleeper's precomputed totals;
    eliminator uses the table above. Projections get no yardage bonus (would
    need P(>=100 yds)), so eliminator projections run a touch low."""
    if not stats:
        return 0.0
    if scoring == "eliminator":
        return eliminator_pts(stats)
    return stats.get("pts_" + scoring) or 0.0


def load_manual(players, projections, stats, games):
    """
    manual.json format:
      [{"name": "Eliminator.gg", "scoring": "eliminator",
        "starters": ["Jalen Hurts", {"name": "Nico Collins", "mult": 1.5}]}]
    scoring is ppr | half_ppr | std | eliminator (see manual_score); mult is
    a per-player multiplier for eliminator-style boosts. Names are matched
    loosely against the Sleeper player map. No opponent, so this is just a
    live score + projected final for your own lineup.
    """
    if not MANUAL_FILE.exists():
        return []
    index = {}
    for pid, p in players.items():
        index.setdefault(norm(p["name"]), pid)

    out = []
    for team in json.loads(MANUAL_FILE.read_text()):
        scoring = team.get("scoring", "ppr")
        resolved, missed, mults = [], [], {}
        for raw in team.get("starters", []):
            name, mult = (raw["name"], raw.get("mult", 1)) if isinstance(raw, dict) else (raw, 1)
            pid = index.get(norm(name))
            if pid:
                resolved.append(pid)
                mults[pid] = mult
            else:
                missed.append(name)
        actual, proj, det = {}, {}, {}
        for pid in resolved:
            act = round(manual_score(stats.get(pid), scoring) * mults[pid], 2)
            pj = round(manual_score(projections.get(pid), scoring) * mults[pid], 2)
            team_code = pid if pid in games else players.get(pid, {}).get("team")
            rem = games.get(team_code, 1.0)
            actual[pid] = act
            proj[pid] = round(act + pj * rem, 2)
            det[pid] = {"act": act, "proj": pj, "rem": rem, "pos": players.get(pid, {}).get("pos", "")}
        out.append({
            "league_id": "manual:" + team["name"],
            "name": team["name"],
            "mode": "manual",
            "starters": resolved,
            "unmatched": missed,
            "my_points": round(sum(actual.values()), 2),
            "my_proj": round(sum(proj.values()), 2),
            "my_to_play": to_play(det),
            "lineup": lineup_rows(det, players, stats),
            "_log": log_rows("me", True, det, players),
            "players_points": actual,
            "starters_proj": proj,
            "mults": {pid: m for pid, m in mults.items() if m != 1},
            # No field to simulate against yet, so the per-point value is a
            # stand-in from manual.json until we have a cut-line model.
            "weight": CONFIG["league_weights"].get("manual:" + team["name"], 1.0),
            "sens": team.get("root_per_pt", 1.0),
        })
    return out


# ----------------------------------------------------------------------------
# ESPN (via espn-api). espn.json:
#   {"espn_s2": "...", "swid": "{...}", "leagues": [{"league_id": 12345}]}
# Cookies come from a logged-in espn.com session (see espn.example.json).
# Optional per-league "team_id" if owner matching by SWID fails.
# ----------------------------------------------------------------------------

ESPN_TEAM_FIX = {"WSH": "WAS"}     # ESPN abbreviations that differ from Sleeper's


def espn_config():
    """Env vars (Fly secrets) win over espn.json. ESPN_LEAGUES is a JSON list
    like [{"league_id": 123}] or just a comma-separated list of ids."""
    if os.environ.get("ESPN_S2"):
        raw = os.environ.get("ESPN_LEAGUES", "")
        try:
            leagues = json.loads(raw) if raw.strip().startswith("[") else \
                [{"league_id": int(x)} for x in raw.split(",") if x.strip()]
        except ValueError:
            leagues = []
        return {"espn_s2": os.environ["ESPN_S2"], "swid": os.environ.get("SWID", ""),
                "leagues": leagues}
    if ESPN_FILE.exists():
        return json.loads(ESPN_FILE.read_text())
    return None


def load_espn(week, players, games, stats):
    """
    Returns (league entries, extra players). Entries match the Sleeper h2h
    shape so exposure needs no changes. Players are mapped onto Sleeper ids
    (espn_id, then loose name) so cross-platform exposure merges; anything
    unmatched keeps an "espn:<id>" key and is described in `extras`.
    """
    cfg = espn_config()
    if not cfg:
        return [], {}
    try:
        from espn_api.football import League
    except ImportError:
        return [{"league_id": "espn", "name": "ESPN", "mode": "error",
                 "error": "pip install espn-api"}], {}

    swid = (cfg.get("swid") or "").strip("{}").lower()
    by_espn = {p["espn_id"]: pid for pid, p in players.items() if p.get("espn_id")}
    by_name = {}
    for pid, p in players.items():
        by_name.setdefault(norm(p["name"]), pid)
    extras = {}

    def resolve(bp):
        team = ESPN_TEAM_FIX.get(bp.proTeam, bp.proTeam)
        if bp.position == "D/ST":
            return team                       # Sleeper DEF ids are team codes
        pid = by_espn.get(bp.playerId) or by_name.get(norm(bp.name))
        if not pid:
            pid = f"espn:{bp.playerId}"
            extras[pid] = {"name": bp.name, "pos": bp.position, "team": team}
        return pid

    slot_order = ["QB", "RB", "WR", "TE", "RB/WR", "WR/TE", "RB/WR/TE", "OP", "D/ST", "K"]

    def side(lineup):
        # ESPN returns roster entries in no useful order; present them by slot.
        starters = sorted((bp for bp in lineup if bp.slot_position not in ("BE", "IR")),
                          key=lambda bp: slot_order.index(bp.slot_position) if bp.slot_position in slot_order else 99)
        ids = [resolve(bp) for bp in starters]
        actual = {resolve(bp): round(bp.points or 0.0, 2) for bp in lineup}
        proj, det = {}, {}
        for pid, bp in zip(ids, starters):
            team = pid if pid in games else players.get(pid, extras.get(pid, {})).get("team")
            rem = games.get(team, 1.0)
            proj[pid] = round((bp.points or 0.0) + (bp.projected_points or 0.0) * rem, 2)
            det[pid] = {"act": bp.points or 0.0, "proj": bp.projected_points or 0.0, "rem": rem,
                        "pos": "DEF" if bp.position == "D/ST" else bp.position}
        return ids, actual, proj, det

    def is_mine(team, lc):
        if lc.get("team_id") is not None:
            return team.team_id == lc["team_id"]
        owners = [(o.get("id") if isinstance(o, dict) else str(o)) or "" for o in (team.owners or [])]
        return any(o.strip("{}").lower() == swid for o in owners)

    out = []
    for lc in cfg.get("leagues", []):
        lid = f"espn:{lc['league_id']}"
        try:
            lg = League(league_id=int(lc["league_id"]), year=int(CONFIG["season"]),
                        espn_s2=cfg.get("espn_s2"), swid=cfg.get("swid"))
            boxes = lg.box_scores(week)
        except Exception as exc:
            out.append({"league_id": lid, "name": lc.get("name") or f"ESPN {lc['league_id']}",
                        "mode": "error", "error": str(exc)})
            continue

        # Whole-league context first, so the drill-down works even for a
        # bye/odd week. Projections use the same blend as our own lineup.
        sides = {}
        for b in boxes:
            for which in ("home", "away"):
                t = getattr(b, f"{which}_team")
                if t:
                    sides[t.team_id] = side(getattr(b, f"{which}_lineup"))
        sims = simulate({rid: d[3] for rid, d in sides.items()}, CONFIG["sims"])

        def box_side(box, which):
            t = getattr(box, f"{which}_team")
            if not t:
                return None
            _, _, proj, det = sides[t.team_id]
            return {"rid": t.team_id, "name": t.team_name,
                    "pts": round(getattr(box, f"{which}_score") or 0.0, 2),
                    "proj": round(sum(proj.values()), 2), "to_play": to_play(det)}
        league_matchups = []
        for b in boxes:
            pair = [x for x in (box_side(b, "home"), box_side(b, "away")) if x]
            if len(pair) == 2:
                pair[0]["win_pct"] = win_pct(sims[pair[0]["rid"]], sims[pair[1]["rid"]])
                pair[1]["win_pct"] = round(100 - pair[0]["win_pct"], 1)
            league_matchups.append(pair)
        standings = sorted(
            [{"rid": t.team_id, "name": t.team_name, "w": t.wins, "l": t.losses, "t": t.ties,
              "pf": round(t.points_for, 2)} for t in lg.teams],
            key=lambda x: (-x["w"], -x["pf"]))

        found = False
        for box in boxes:
            for me, them in (("home", "away"), ("away", "home")):
                team = getattr(box, f"{me}_team")
                if not team or not is_mine(team, lc):
                    continue
                found = True
                my_ids, my_actual, my_proj, my_det = sides[team.team_id]
                entry = {
                    "league_id": lid, "source": "espn",
                    "name": lc.get("name") or lg.settings.name,
                    "week": week, "mode": "h2h", "best_ball": False,
                    "weight": CONFIG["league_weights"].get(lid, 1.0),
                    "my_points": round(getattr(box, f"{me}_score") or 0.0, 2),
                    "my_proj": round(sum(my_proj.values()), 2),
                    "my_to_play": to_play(my_det),
                    "lineup": lineup_rows(my_det, {**players, **extras}, stats),
                    "_log": [r for rid, d in sides.items()
                             for r in log_rows(rid, rid == team.team_id, d[3], {**players, **extras})],
                    "starters": my_ids, "starters_proj": my_proj,
                    "players_points": my_actual,
                    "matchups": league_matchups, "standings": standings, "my_rid": team.team_id,
                }
                opp = getattr(box, f"{them}_team")
                if opp:
                    opp_ids, opp_actual, opp_proj, opp_det = sides[opp.team_id]
                    entry.update({
                        "opp_name": opp.team_name,
                        "opp_to_play": to_play(opp_det),
                        "opp_lineup": lineup_rows(opp_det, {**players, **extras}, stats),
                        "win_pct": win_pct(sims[team.team_id], sims[opp.team_id]),
                        "sens": point_value(sims[team.team_id], sims[opp.team_id]),
                        "opp_points": round(getattr(box, f"{them}_score") or 0.0, 2),
                        "opp_starters": opp_ids, "opp_players_points": opp_actual,
                        "opp_proj": round(sum(opp_proj.values()), 2),
                        "opp_starters_proj": opp_proj,
                    })
                    entry["margin"] = round(entry["my_points"] - entry["opp_points"], 2)
                    entry["proj_margin"] = round(entry["my_proj"] - entry["opp_proj"], 2)
                out.append(entry)
        if not found:
            names = ", ".join(f"{t.team_id}: {t.team_name}" for t in lg.teams)
            out.append({"league_id": lid, "name": lc.get("name") or lg.settings.name, "mode": "error",
                        "error": f"couldn't find your team by SWID; set \"team_id\" in espn.json to one of {names}"})
    return out, extras


# ----------------------------------------------------------------------------
# Calibration log. Every snapshot: each starter in every lineup we can see
# (all rosters, not just mine), with projection, actual and time left, plus
# a per-league summary. Enough to fit real position spreads and the
# eliminator distribution after a few weeks.
# ----------------------------------------------------------------------------

def log_snapshot(season, week, leagues):
    if not CONFIG["log_seconds"] or time.time() - _log_last["ts"] < CONFIG["log_seconds"]:
        return
    ts = int(time.time())
    con = sqlite3.connect(LOG_DB)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            ts INTEGER, season TEXT, week INTEGER, league_id TEXT, league TEXT,
            rid TEXT, mine INTEGER, pid TEXT, name TEXT, pos TEXT, team TEXT,
            act REAL, proj REAL, rem REAL);
        CREATE INDEX IF NOT EXISTS players_wk ON players (season, week, pid);
        CREATE TABLE IF NOT EXISTS leagues (
            ts INTEGER, season TEXT, week INTEGER, league_id TEXT, league TEXT, mode TEXT,
            my_points REAL, my_proj REAL, pct REAL, sens REAL, rank INTEGER, field_size INTEGER,
            to_play INTEGER, live INTEGER);
    """)
    prow, lrow = [], []
    for lg in leagues:
        if lg["mode"] == "error":
            continue
        for r in lg.pop("_log", []):
            prow.append((ts, season, week, lg["league_id"], lg["name"], *r))
        tp = lg.get("my_to_play") or (None, None)
        lrow.append((ts, season, week, lg["league_id"], lg["name"], lg["mode"],
                     lg.get("my_points"), lg.get("my_proj"),
                     lg.get("survive_pct", lg.get("win_pct")), lg.get("sens"),
                     lg.get("rank"), lg.get("field_size"), tp[0], tp[1]))
    con.executemany("INSERT INTO players VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", prow)
    con.executemany("INSERT INTO leagues VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", lrow)
    con.commit()
    con.close()
    _log_last["ts"] = time.time()


def log_rows(rid, mine, det, players):
    return [(str(rid), int(mine), pid, players.get(pid, {}).get("name", pid),
             players.get(pid, {}).get("pos", d.get("pos", "")), players.get(pid, {}).get("team", ""),
             round(d["act"], 2), round(d["proj"], 2), round(d["rem"], 3))
            for pid, d in det.items()]


def shared_pool_view(entry, field, pfield, sims, proj_by_rid, players, stats):
    """
    Every team's seat in a guillotine league, each shaped like our own pool
    entry so the page can reuse chopCard/poolDetail unchanged.
    """
    n = len(field)
    live_order = [f["rid"] for f in field]
    proj_order = [f["rid"] for f in pfield]
    teams = []
    for f in field:
        rid = f["rid"]
        li, pi = live_order.index(rid), proj_order.index(rid)
        margin, ref = chop_line(field, li, "pts")
        pmargin, pref = chop_line(pfield, pi, "proj")
        others = [sims[x] for x in live_order if x != rid]
        _, _, det = proj_by_rid[rid]
        teams.append({
            "league_id": entry["league_id"], "name": entry["name"], "mode": "pool",
            "best_ball": entry["best_ball"], "week": entry["week"],
            "rid": rid, "team": f["name"], "my_rid": rid,
            "my_points": f["pts"], "my_proj": f["proj"], "my_to_play": f["to_play"],
            "field": field, "field_size": n,
            "rank": n - li, "margin": margin, "ref": ref,
            "below": field[li - 1] if li > 0 else None,
            "above": field[li + 1] if li + 1 < n else None,
            "proj_rank": n - pi, "proj_margin": pmargin, "proj_ref": pref,
            "survive_pct": round(100 - f["chop_pct"], 1),
            "sens": point_value(sims[rid], np.min(others, axis=0)),
            "lineup": lineup_rows(det, players, stats),
            "ref_lineup": lineup_rows(proj_by_rid[pref["rid"]][2], players, stats),
        })
    roster = {}
    for t in teams:
        for r in t["lineup"]:
            roster.setdefault(r["pid"], {"id": r["pid"], "name": r["name"], "team": r["team"],
                                         "pts": r["act"], "for": [], "against": []})
    return {"teams": teams, "feed": build_feed(list(roster.values()))}


# ----------------------------------------------------------------------------
# Build the picture
# ----------------------------------------------------------------------------

def build():
    players = load_players()
    state = get("/state/nfl")
    week = state.get("week") or 1
    projections = load_projections(CONFIG["season"], week)
    games = load_game_state(CONFIG["season"], week)
    stats = load_stats(CONFIG["season"], week)

    user = get(f"/user/{CONFIG['sleeper_username']}")
    uid = user["user_id"]
    leagues = get(f"/user/{uid}/leagues/nfl/{CONFIG['season']}")

    out_leagues = []
    # player_id -> {"for": [league names], "against": [league names], "bench": [...]}
    exposure = {}

    for lg in leagues:
        lid = lg["league_id"]
        if lid in CONFIG["exclude_leagues"]:
            continue

        rosters = get(f"/league/{lid}/rosters")
        users = {u["user_id"]: (u.get("display_name") or "?") for u in get(f"/league/{lid}/users")}
        matchups = get(f"/league/{lid}/matchups/{week}")
        by_roster = {m["roster_id"]: m for m in matchups}

        mine = next((r for r in rosters if r.get("owner_id") == uid), None)
        if mine is None:
            continue
        my_rid = mine["roster_id"]
        my_m = by_roster.get(my_rid, {})

        # Pool/guillotine detection. Sleeper tags guillotine leagues with
        # settings.type == 3, and also hands every roster its own synthetic
        # matchup_id, so "no matchup_id is shared by two rosters" is the
        # structural tell. Head-to-head leagues pair rosters (N rosters, N/2 ids).
        ids = [m.get("matchup_id") for m in matchups if m.get("matchup_id") is not None]
        is_pool = (
            lid in CONFIG["force_pool_leagues"]
            or lg.get("settings", {}).get("type") == 3
            or len(ids) == len(set(ids))
        )

        scoring = lg.get("scoring_settings") or {}
        best_ball = bool((lg.get("settings") or {}).get("best_ball"))

        def team_name(rid):
            r = next((x for x in rosters if x["roster_id"] == rid), None)
            return users.get(r.get("owner_id")) if r else f"Roster {rid}"

        def lineup(m):
            """
            Who counts for this roster. Sleeper's best-ball `starters` is a
            stale auto-lineup pregame, so there we pick the best lineup by
            projected final over the whole roster instead.
            """
            actual = m.get("players_points") or {}
            val = lambda p: blended(p, actual, projections, scoring, players, games)
            if best_ball:
                ids = best_lineup(m.get("players") or [], lg.get("roster_positions") or [], val, players)
            else:
                ids = [p for p in (m.get("starters") or []) if p and p != "0"]
            det = {}
            for p in ids:
                team = p if p in games else players.get(p, {}).get("team")
                det[p] = {"act": actual.get(p) or 0.0, "proj": proj_pts(p, projections, scoring) or 0.0,
                          "rem": games.get(team, 1.0), "pos": players.get(p, {}).get("pos", "")}
            return ids, {p: val(p) for p in ids}, det

        proj_by_rid = {m["roster_id"]: lineup(m) for m in matchups}
        sims = simulate({rid: det for rid, (_, _, det) in proj_by_rid.items()}, CONFIG["sims"])
        field = sorted(
            [{"rid": m["roster_id"], "name": team_name(m["roster_id"]),
              "pts": round(m.get("points") or 0.0, 2),
              "proj": round(sum(proj_by_rid[m["roster_id"]][1].values()), 2),
              "to_play": to_play(proj_by_rid[m["roster_id"]][2])}
             for m in matchups],
            key=lambda x: x["pts"],
        )
        my_lineup, my_proj, my_det = proj_by_rid.get(my_rid, ([], {}, {}))

        entry = {
            "league_id": lid,
            "name": lg.get("name", lid),
            "week": week,
            "mode": "pool" if is_pool else "h2h",
            "my_points": round(my_m.get("points") or 0.0, 2),
            "best_ball": best_ball,
            "weight": CONFIG["league_weights"].get(lid, 1.0),
            "my_proj": round(sum(my_proj.values()), 2),
            "my_to_play": to_play(my_det),
            "lineup": lineup_rows(my_det, players, stats),
            "_log": [r for rid, (_, _, det) in proj_by_rid.items()
                     for r in log_rows(rid, rid == my_rid, det, players)],
            # Best ball: the rest of the roster can still play its way into
            # the lineup, so the feed watches them too (tagged bench).
            "bench": [{"id": p, "pts": round((my_m.get("players_points") or {}).get(p) or 0.0, 2)}
                      for p in (my_m.get("players") or []) if best_ball and p not in my_lineup],
            "starters": my_lineup,
            "starters_proj": my_proj,
            "starters_points": my_m.get("starters_points") or [],
            "players_points": my_m.get("players_points") or {},
        }

        if is_pool:
            order = [f["rid"] for f in field]
            rank = order.index(my_rid)          # 0 = currently chopped
            margin, ref = chop_line(field, rank, "pts")
            entry.update({
                "field": field,
                "rank": len(field) - rank,          # 1 = top scorer, N = on the block
                "field_size": len(field),
                "my_rid": my_rid,
                "margin": margin,
                "ref": ref,
                "below": field[rank - 1] if rank > 0 else None,
                "above": field[rank + 1] if rank + 1 < len(field) else None,
                "chop_target": field[0],
            })
            # Same picture on projections: where the chop line lands if
            # everyone hits their number.
            pfield = sorted(field, key=lambda x: x["proj"])
            prank = [f["rid"] for f in pfield].index(my_rid)
            pmargin, pref = chop_line(pfield, prank, "proj")
            # Chop probability: share of sims in which each team is the low score.
            mat = np.array([sims[f["rid"]] for f in field])
            share = np.bincount(mat.argmin(axis=0), minlength=len(field)) / mat.shape[1]
            for f, c in zip(field, share):
                f["chop_pct"] = round(100 * float(c), 1)
            entry.update({
                "proj_rank": len(field) - prank,
                "proj_margin": pmargin,
                "proj_ref": pref,
                "proj_chop_target": pfield[0],
                "survive_pct": round(100 - field[rank]["chop_pct"], 1),
                "ref_lineup": lineup_rows(proj_by_rid[pref["rid"]][2], players, stats),
            })
            if lid in CONFIG["shared_leagues"]:
                entry["shared"] = shared_pool_view(entry, field, pfield, sims, proj_by_rid, players, stats)

            others = {f["rid"]: sims[f["rid"]] for f in field if f["rid"] != my_rid}
            entry["sens"] = point_value(sims[my_rid], np.min(list(others.values()), axis=0))
            # The nearest rival's starters count against you, scaled by how
            # much their scoring actually threatens you.
            rest = [a for rid, a in others.items() if rid != pref["rid"]]
            ref_ids, ref_proj, ref_det = proj_by_rid[pref["rid"]]
            entry.update({
                "ref_sens": rival_point_value(sims[my_rid], sims[pref["rid"]],
                                              np.min(rest, axis=0) if rest else np.full_like(sims[my_rid], np.inf)),
                "opp_name": pref["name"],
                "opp_starters": ref_ids,
                "opp_starters_proj": ref_proj,
                "opp_players_points": {p: d["act"] for p, d in ref_det.items()},
            })
        else:
            mid = my_m.get("matchup_id")
            opp_m = next((m for m in matchups
                          if m.get("matchup_id") == mid and m["roster_id"] != my_rid), None)
            # Whole-league context for the drill-down: every matchup this
            # week and the standings (Sleeper keeps W-L / PF on the roster).
            by_mid = {}
            for m in matchups:
                by_mid.setdefault(m.get("matchup_id"), []).append(m)
            side = lambda m: {"rid": m["roster_id"], "name": team_name(m["roster_id"]),
                              "pts": round(m.get("points") or 0.0, 2),
                              "proj": round(sum(proj_by_rid[m["roster_id"]][1].values()), 2),
                              "to_play": to_play(proj_by_rid[m["roster_id"]][2])}
            entry["matchups"] = []
            for k, pair in by_mid.items():
                if k is None:
                    continue
                pair = [side(x) for x in pair]
                if len(pair) == 2:
                    pair[0]["win_pct"] = win_pct(sims[pair[0]["rid"]], sims[pair[1]["rid"]])
                    pair[1]["win_pct"] = round(100 - pair[0]["win_pct"], 1)
                entry["matchups"].append(pair)
            entry["standings"] = sorted(
                [{"rid": r["roster_id"], "name": team_name(r["roster_id"]),
                  "w": (r.get("settings") or {}).get("wins", 0),
                  "l": (r.get("settings") or {}).get("losses", 0),
                  "t": (r.get("settings") or {}).get("ties", 0),
                  "pf": round((r.get("settings") or {}).get("fpts", 0)
                              + (r.get("settings") or {}).get("fpts_decimal", 0) / 100, 2)}
                 for r in rosters],
                key=lambda x: (-x["w"], -x["pf"]))
            entry["my_rid"] = my_rid
            if opp_m:
                entry["opp_name"] = team_name(opp_m["roster_id"])
                entry["opp_points"] = round(opp_m.get("points") or 0.0, 2)
                opp_lineup, opp_proj, opp_det = proj_by_rid[opp_m["roster_id"]]
                entry["opp_starters"] = opp_lineup
                entry["opp_to_play"] = to_play(opp_det)
                entry["opp_lineup"] = lineup_rows(opp_det, players, stats)
                entry["win_pct"] = win_pct(sims[my_rid], sims[opp_m["roster_id"]])
                entry["sens"] = point_value(sims[my_rid], sims[opp_m["roster_id"]])
                entry["opp_players_points"] = opp_m.get("players_points") or {}
                entry["opp_proj"] = round(sum(opp_proj.values()), 2)
                entry["opp_starters_proj"] = opp_proj
                entry["margin"] = round(entry["my_points"] - entry["opp_points"], 2)
                entry["proj_margin"] = round(entry["my_proj"] - entry["opp_proj"], 2)

        out_leagues.append(entry)

    # Manual teams join the exposure book but have no live scoring of their own.
    out_leagues.extend(load_manual(players, projections, stats, games))

    espn_leagues, extras = load_espn(week, players, games, stats)
    out_leagues.extend(espn_leagues)
    players = {**players, **extras}

    # --- exposure ---
    # root: sum over leagues of +-weight x per-point value (pp of survival /
    # win per fantasy point). impact: root x remaining projection, i.e. the
    # swing this player still has in him tonight.
    for lg in out_leagues:
        if lg["mode"] == "error":
            continue
        short = lg["name"][:14]
        w = lg.get("weight", 1.0)
        # Per-point value of my starters, and of the other side's: the h2h
        # opponent is a mirror image; a pool rival has its own (negative) figure.
        mine = w * lg.get("sens", 0.0)
        theirs = w * lg.get("ref_sens", 0.0) if lg["mode"] == "pool" else -mine
        for side, sign, per_pt in (("", 1, mine), ("opp_", -1, theirs)):
            pp = lg.get(side + "players_points", {})
            sp = lg.get(side + "starters_proj", {})
            for pid in lg.get(side + "starters", []):
                if not pid or pid == "0":
                    continue
                e = exposure.setdefault(pid, {"for": [], "against": [], "pts": None, "proj": None,
                                              "root": 0.0, "impact": 0.0, "leagues": []})
                e["for" if sign > 0 else "against"].append(short)
                e["leagues"].append({
                    "name": lg["name"], "side": "for" if sign > 0 else "against",
                    "pts": pp.get(pid), "final": sp.get(pid), "per_pt": round(per_pt, 2),
                    "impact": round(per_pt * max(0.0, (sp.get(pid) or 0.0) - (pp.get(pid) or 0.0)), 1),
                    "weight": w,
                })
                if pid in pp and (e["pts"] is None or sign > 0):
                    e["pts"] = round(pp[pid], 2)
                if e["proj"] is None or sign > 0:
                    e["proj"] = sp.get(pid, e["proj"])
                remaining = max(0.0, (sp.get(pid) or 0.0) - (pp.get(pid) or 0.0))
                e["root"] += per_pt
                e["impact"] += per_pt * remaining

    book = []
    for pid, e in exposure.items():
        p = players.get(pid, {"name": pid, "pos": "", "team": ""})
        net = len(e["for"]) - len(e["against"])
        team = pid if pid in games else p.get("team")
        book.append({
            "line": stat_line(stats.get(pid), p.get("pos", "")),
            "rem": round(games.get(team, 1.0), 3),
            "ot": team in OT_TEAMS,
            "leagues": e["leagues"],
            "id": pid, "name": p["name"], "pos": p["pos"], "team": p["team"],
            "pts": e["pts"], "proj": e["proj"], "for": e["for"], "against": e["against"],
            "n_for": len(e["for"]), "n_against": len(e["against"]), "net": net,
            "root": round(e["root"], 2), "impact": round(e["impact"], 1),
        })
    # Sort by what is still at stake tonight, then by how much each point matters.
    book.sort(key=lambda x: (-abs(x["impact"]), -abs(x["root"])))

    watch = {p["id"]: dict(p) for p in book}
    for lg in out_leagues:
        for b in lg.get("bench", []):
            tag = f"{lg['name'][:14]} (bench)"
            if b["id"] in watch:
                watch[b["id"]] = {**watch[b["id"]], "for": watch[b["id"]]["for"] + [tag]}
            else:
                pl = players.get(b["id"], {})
                watch[b["id"]] = {"id": b["id"], "name": pl.get("name", b["id"]), "team": pl.get("team", ""),
                                  "pts": b["pts"], "for": [tag], "against": []}
    feed = build_feed(list(watch.values()), players)
    log_snapshot(CONFIG["season"], week, out_leagues)   # pops the _log rows
    for lg in out_leagues:
        lg.pop("_log", None)

    return {"week": week, "leagues": out_leagues, "book": book, "feed": feed,
            "updated": time.strftime("%H:%M:%S")}


@app.before_request
def basic_auth():
    """Single shared password via BLOTTER_PASSWORD; open when unset (local)."""
    pw = os.environ.get("BLOTTER_PASSWORD")
    if not pw or request.path == "/healthz" or request.path.startswith(("/l/", "/api/league/")):
        return None
    auth = request.headers.get("Authorization", "")
    ok = False
    if auth.startswith("Basic "):
        try:
            _, _, got = base64.b64decode(auth[6:]).decode().partition(":")
            ok = got == pw
        except Exception:
            ok = False
    if not ok:
        return Response("auth required", 401, {"WWW-Authenticate": 'Basic realm="blotter"'})


def current_state():
    """Cached build, refreshed at most every poll_seconds. Returns (data, error)."""
    with _build_lock:
        now = time.time()
        if _cache["data"] is None or now - _cache["ts"] > CONFIG["poll_seconds"]:
            try:
                _cache["data"] = build()
                _cache["ts"] = now
            except Exception as exc:
                if _cache["data"] is None:
                    return None, str(exc)
                _cache["data"]["stale"] = str(exc)
        return _cache["data"], None


@app.route("/api/state")
def api_state():
    data, err = current_state()
    if err:
        return jsonify({"error": err}), 502
    return jsonify(data)


@app.route("/api/league/<lid>")
def api_league(lid):
    """Public per-team view of a shared guillotine league."""
    if lid not in CONFIG["shared_leagues"]:
        return jsonify({"error": "not shared"}), 404
    data, err = current_state()
    if err:
        return jsonify({"error": err}), 502
    lg = next((l for l in data["leagues"] if l["league_id"] == lid and l.get("shared")), None)
    if not lg:
        return jsonify({"error": "league not found"}), 404
    return jsonify({"week": data["week"], "updated": data["updated"], "stale": data.get("stale"),
                    "name": lg["name"], "teams": lg["shared"]["teams"], "feed": lg["shared"]["feed"]})


@app.route("/l/<lid>")
def league_page(lid):
    if lid not in CONFIG["shared_leagues"]:
        return Response("not shared", 404)
    return Response(PAGE.replace("<script>", f"<script>const LEAGUE_ID = {json.dumps(lid)};", 1),
                    mimetype="text/html")


@app.route("/healthz")
def healthz():
    return "ok"


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fantasydash</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#E4E7E1; --panel:#EEF0EB; --ink:#1F2621; --mute:#6E756C;
    --rule:#C6CBC2; --long:#1F6F4A; --short:#A6402B; --warn:#B07A16;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.45 "IBM Plex Sans",system-ui,sans-serif;
       padding:20px 18px 60px;max-width:860px;margin:0 auto}
  .num{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
  h1{font-size:15px;font-weight:600;margin:0;letter-spacing:.01em}
  header{display:flex;justify-content:space-between;align-items:baseline;
         border-bottom:1.5px solid var(--ink);padding-bottom:8px;margin-bottom:16px}
  #app > h2:first-child{margin-top:0}
  header .meta{font-size:12.5px;color:var(--mute)}
  h2{font-size:13px;font-weight:600;color:var(--mute);margin:30px 0 10px}
  h2.sec{cursor:pointer;user-select:none}
  h2.sec:hover{color:var(--ink)}
  h2 .caret{display:inline-block;width:12px;font-size:11px}
  h2 .cnt{font-weight:400;opacity:.7;margin-left:4px}

  /* chop-line hero */
  .chop{background:var(--panel);border-left:3px solid var(--ink);
        padding:14px 16px;margin-bottom:10px}
  .chop .lg{font-size:13px;color:var(--mute);margin-bottom:6px}
  .chop .lg b{font-weight:600;color:var(--ink)}
  .chop .big{font-size:40px;font-weight:600;line-height:1;letter-spacing:-.02em}
  .chop .big small{font-size:14px;font-weight:400;color:var(--mute);margin-left:8px;
                   letter-spacing:0}
  .chop.danger{border-left-color:var(--short)}
  .chop.danger .big{color:var(--short)}
  .chop.thin{border-left-color:var(--warn)}
  .neighbors{display:flex;gap:26px;margin-top:12px;font-size:13px;color:var(--mute)}
  .neighbors b{font-weight:500;color:var(--ink)}
  .chop .proj{margin-top:8px;font-size:12.5px;color:var(--mute)}

  table{width:100%;border-collapse:collapse;font-size:14px}
  th{text-align:left;font-weight:500;font-size:12.5px;color:var(--mute);
     border-bottom:1px solid var(--rule);padding:0 8px 6px 0}
  td{padding:6px 8px 6px 0;border-bottom:1px solid var(--rule);vertical-align:top}
  td.r,th.r{text-align:right;padding-right:0}
  .pos{color:var(--mute);font-size:12.5px}
  .tags{font-size:12px;color:var(--mute);padding-left:18px}
  tr.done td{color:var(--mute)}
  tr.prow{cursor:pointer}
  tr.prow:hover td{background:var(--panel)}
  tr.pdetail td{padding:0;border-bottom:none}
  .long{color:var(--long);font-weight:500}
  .short{color:var(--short);font-weight:500}
  .row-h2h{display:flex;justify-content:space-between;padding:8px 0;
           border-bottom:1px solid var(--rule);font-size:14px}
  .err{background:var(--short);color:#fff;padding:10px 14px;margin-bottom:16px}
  .chop,.row-h2h{cursor:pointer}
  .chop:hover,.row-h2h:hover{background:var(--panel)}
  .detail{background:var(--panel);padding:10px 16px 14px;margin:-6px 0 12px;
          border-left:3px solid var(--rule);font-size:13px}
  .detail h3{font-size:12px;font-weight:600;color:var(--mute);margin:8px 0 6px}
  .detail table{font-size:13px}
  .detail td,.detail th{padding:3px 8px 3px 0}
  .detail tr.me td{font-weight:600}
  .detail tr.line td{border-bottom:2px solid var(--short)}
  .detail .rk{color:var(--mute);width:2.2em}
  .detail table.lineup td{white-space:nowrap}
  .detail table.lineup td:nth-child(3){white-space:normal;font-size:12px;min-width:160px}
  .detail h3 .num{color:var(--ink)}
  .fev{display:grid;grid-template-columns:52px 1.1fr 2.4fr 1fr 1fr;gap:10px;align-items:baseline;
       padding:6px 0;border-bottom:1px solid var(--rule);font-size:13.5px}
  .fev.score .fwhat{color:var(--ink);font-weight:500}
  .fev.score{background:var(--panel)}
  .fev.fhead{font-size:12px;color:var(--mute);font-weight:500}
  .fev .fd{text-align:right}
  .fev .tags{padding-left:0;font-weight:400}
  .fev .fwhat{font-size:12.5px}
  .pick{margin:0 0 18px;font-size:13px;color:var(--mute)}
  .pick select{font:inherit;padding:4px 8px;border:1px solid var(--rule);background:var(--panel);color:var(--ink)}
  @media (prefers-reduced-motion:no-preference){
    .tick{transition:color .4s ease}
  }
</style>
<header>
  <h1 id="h1">fantasydash</h1>
  <div class="meta num" id="meta">loading</div>
</header>
<div id="app"></div>
<script>
const $ = s => document.querySelector(s);
if (typeof LEAGUE_ID === 'undefined') window.LEAGUE_ID = null;
const open = new Set();   // league ids with the drill-down expanded; survives re-render
let showMinor = false;    // feed: show the full list instead of the last 12
const collapsed = new Set((() => { try { return JSON.parse(localStorage.getItem('collapsed') || '[]'); } catch(e){ return []; } })());
function toggleSection(id){
  collapsed.has(id) ? collapsed.delete(id) : collapsed.add(id);
  try { localStorage.setItem('collapsed', JSON.stringify([...collapsed])); } catch(e){}
  tick();
}
function section(id, title, body, count){
  const c = collapsed.has(id);
  return `<h2 class="sec" onclick="toggleSection('${id}')"><span class="caret">${c ? '▸' : '▾'}</span> ${title}${count !== undefined ? ` <span class="cnt">${count}</span>` : ''}</h2>${c ? '' : body}`;
}
function toggle(id){ open.has(id) ? open.delete(id) : open.add(id); tick(); }
const f2 = x => (x ?? 0).toFixed(2);
const tp = t => t ? `${t[0]} to play${t[1] ? ` (${t[1]} live)` : ''}` : '';
const pctCls = p => p >= 85 ? 'long' : (p < 60 ? 'short' : '');

function lineupTable(title, rows){
  if (!rows || !rows.length) return '';
  const left = r => r.rem >= 1 ? '<span class="pos">—</span>' : (r.rem <= 0 ? '<span class="pos">final</span>' : `<span style="color:var(--warn)">${Math.round(r.rem*60)}'${r.ot ? ' OT' : ''}</span>`);
  const tot = k => rows.reduce((a,r) => a + r[k], 0);
  return `<h3>${title} &middot; <span class="num">${f2(tot('act'))}</span> &rarr; <span class="num">${f2(tot('final'))}</span></h3>
    <table class="lineup"><tr><th>Pos</th><th>Player</th><th>Line</th><th class="r">Pts</th><th class="r">Proj final</th><th class="r">Left</th></tr>
    ${rows.map(r => `<tr>
      <td class="pos">${r.pos}</td><td>${r.name} <span class="pos">${r.team}</span></td>
      <td class="pos">${r.line}</td>
      <td class="r num">${r.act ? f2(r.act) : '<span class="pos">0.00</span>'}</td>
      <td class="r num ${r.final > r.proj ? 'long' : (r.rem <= 0 && r.final < r.proj ? 'short' : '')}">${f2(r.final)}<span class="pos" style="font-size:11px"> /${r.proj.toFixed(1)}</span></td>
      <td class="r num">${left(r)}</td></tr>`).join('')}
    </table>`;
}

function poolDetail(l){
  // Field sorted by projected final, high -> low: the projected chop is the
  // last row, line drawn above it. Live rank alongside in grey (l.field is
  // in live order, low -> high).
  const n = l.field.length;
  const lrank = Object.fromEntries(l.field.map((t,i) => [t.rid, n - i]));
  const me = l.my_rid;
  const proj = [...l.field].sort((a,b) => b.proj - a.proj);
  return `<div class="detail">
    <h3>Field · by proj final (chop line above the last row) · live rank in grey</h3>
    <table><tr><th class="rk">#</th><th>Team</th><th class="r">Chop %</th><th class="r">Proj final</th><th class="r">Pts</th><th class="r">To play</th><th class="r">Live #</th></tr>
    ${proj.map((t,i) => `<tr class="${t.rid === me ? 'me' : ''} ${i === n - 2 ? 'line' : ''}">
      <td class="rk num">${i+1}</td><td>${t.name}</td>
      <td class="r num ${t.chop_pct >= 15 ? 'short' : ''}">${Math.round(t.chop_pct)}</td>
      <td class="r num">${f2(t.proj)}</td><td class="r num">${f2(t.pts)}</td>
      <td class="r num pos">${t.to_play[0]}${t.to_play[1] ? `<span style="color:var(--warn)"> ·${t.to_play[1]}</span>` : ''}</td>
      <td class="r num pos">${lrank[t.rid]}</td></tr>`).join('')}
    </table>
    ${lineupTable('Your lineup', l.lineup)}
    ${lineupTable(`${l.proj_ref.name}`, l.ref_lineup)}
    </div>`;
}

function h2hDetail(l){
  const me = l.my_rid;
  const row = (a,b) => {
    const lead = a.pts - b.pts, plead = a.proj - b.proj;
    const cls = x => x > 0 ? 'long' : (x < 0 ? 'short' : '');
    return `<tr class="${a.rid === me || b.rid === me ? 'me' : ''}">
      <td>${a.name}</td><td class="r num ${cls(lead)}">${f2(a.pts)}</td><td class="r num pos">${f2(a.proj)}</td>
      <td class="pos" style="padding:3px 10px">v</td>
      <td>${b.name}</td><td class="r num ${cls(-lead)}">${f2(b.pts)}</td><td class="r num pos">${f2(b.proj)}</td>
      <td class="r num pos">${a.win_pct === undefined ? '' : `${Math.round(a.win_pct)}%`}</td></tr>`;
  };
  return `<div class="detail">
    ${lineupTable('Your lineup', l.lineup)}
    ${lineupTable(l.opp_name ?? 'Opponent', l.opp_lineup)}
    <h3>This week · pts, proj final · win % for the left team</h3>
    <table>${(l.matchups || []).map(p => p.length === 2 ? row(p[0], p[1]) : '').join('')}</table>
    <h3>Standings</h3>
    <table><tr><th class="rk">#</th><th>Team</th><th class="r">W-L</th><th class="r">PF</th></tr>
    ${(l.standings || []).map((t,i) => `<tr class="${t.rid === me ? 'me' : ''}">
      <td class="rk num">${i+1}</td><td>${t.name}</td>
      <td class="r num">${t.w}-${t.l}${t.t ? '-' + t.t : ''}</td><td class="r num">${f2(t.pf)}</td></tr>`).join('')}
    </table></div>`;
}

function chopCard(l){
  const m = l.margin, size = l.field_size;
  const cls = m <= 0 ? 'danger' : (m < 8 ? 'thin' : '');
  const verdict = l.rank === size ? `on the block, ${(-m).toFixed(2)} behind ${l.ref.name}` : (m <= 0 ? 'tied at the line' : 'above the line');
  const pm = l.proj_margin, pr = l.proj_ref;
  const projLine = l.proj_rank === size
    ? `<span class="short">${pm.toFixed(2)}</span> behind ${pr.name} (${pr.proj.toFixed(2)})`
    : `<span class="${pm < 8 ? 'short' : 'long'}">+${pm.toFixed(2)}</span> over ${pr.name} (${pr.proj.toFixed(2)})`;
  const nb = [];
  if (l.below) nb.push(`<span>below &nbsp;<b>${l.below.name}</b> <span class="num">${l.below.pts}</span></span>`);
  if (l.above) nb.push(`<span>above &nbsp;<b>${l.above.name}</b> <span class="num">${l.above.pts}</span></span>`);
  return `<div class="chop ${cls}" onclick="toggle('${l.league_id}')">
    <div class="lg">${l.name}${l.best_ball ? ' &middot; best ball' : ''} &middot; proj <b>${l.proj_rank} of ${size}</b> <span style="opacity:.7">&middot; live ${l.rank}</span></div>
    <div class="big num tick">${m > 0 ? '+' : ''}${m.toFixed(2)}<small>${verdict} &middot; you ${l.my_points.toFixed(2)}</small></div>
    <div class="neighbors">${nb.join('')}</div>
    <div class="proj num">proj final ${l.my_proj.toFixed(2)} &middot; ${projLine}</div>
    <div class="proj num"><span class="${pctCls(l.survive_pct)}" style="font-size:15px">survive ${Math.round(l.survive_pct)}%</span> &middot; ${l.sens.toFixed(2)}%/pt &middot; you ${tp(l.my_to_play)} &middot; ${pr.name} ${tp(pr.to_play)}</div>
  </div>${open.has(l.league_id) ? poolDetail(l) : ''}`;
}

function h2hRow(l){
  const m = l.margin ?? 0, pm = l.proj_margin ?? null;
  const c = m >= 0 ? 'long' : 'short';
  return `<div class="row-h2h" onclick="toggle('${l.league_id}')">
    <span>${l.name} <span class="pos">vs ${l.opp_name ?? '—'}${l.best_ball ? ' · best ball' : ''}${l.source === 'espn' ? ' · espn' : ''}</span></span>
    <span class="num"><span class="${c}">${m >= 0 ? '+' : ''}${m.toFixed(2)}</span>
      &nbsp;<span class="pos">${l.my_points.toFixed(2)} – ${(l.opp_points ?? 0).toFixed(2)}</span>
      ${pm === null ? '' : `&nbsp;<span class="pos">· proj final <span class="${pm >= 0 ? 'long' : 'short'}">${pm >= 0 ? '+' : ''}${pm.toFixed(2)}</span> (${l.my_proj.toFixed(2)} – ${l.opp_proj.toFixed(2)})</span>`}
      ${l.win_pct === undefined ? '' : `&nbsp;<span class="${pctCls(l.win_pct)}">win ${Math.round(l.win_pct)}%</span> <span class="pos">· ${l.sens.toFixed(2)}%/pt${l.weight !== 1 ? ` · w${l.weight}` : ''} · ${l.my_to_play[0]} v ${l.opp_to_play[0]} to play</span>`}</span>
  </div>${open.has(l.league_id) ? h2hDetail(l) : ''}`;
}

function manualRow(l){
  return `<div class="row-h2h" onclick="toggle('${l.league_id}')">
    <span>${l.name}</span>
    <span class="num">${f2(l.my_points)} <span class="pos">· proj final ${f2(l.my_proj)} · ${tp(l.my_to_play)}</span></span>
  </div>${open.has(l.league_id) ? `<div class="detail">${lineupTable('Your lineup', l.lineup)}</div>` : ''}`;
}

function playerDetail(p){
  const left = p.rem >= 1 ? 'not started' : (p.rem <= 0 ? 'final' : `${Math.round(p.rem*60)}'${p.ot ? ' OT' : ''} left`);
  const sgn = x => x > 0 ? 'long' : (x < 0 ? 'short' : 'pos');
  return `<tr class="pdetail"><td colspan="6"><div class="detail" style="margin:4px 0 8px">
    <div class="num" style="margin-bottom:8px"><b>${p.line || 'no stats yet'}</b> <span class="pos">· ${left}</span></div>
    <table><tr><th>League</th><th>Side</th><th class="r">Pts</th><th class="r">Proj final</th><th class="r">Root /pt</th><th class="r">Impact</th></tr>
    ${p.leagues.map(l => `<tr>
      <td>${l.name}${l.weight !== 1 ? ` <span class="pos">w${l.weight}</span>` : ''}</td>
      <td class="${l.side === 'for' ? 'long' : 'short'}">${l.side === 'for' ? 'yours' : 'against'}</td>
      <td class="r num">${l.pts == null ? '<span class="pos">—</span>' : f2(l.pts)}</td>
      <td class="r num pos">${l.final == null ? '—' : f2(l.final)}</td>
      <td class="r num ${sgn(l.per_pt)}">${l.per_pt > 0 ? '+' : ''}${l.per_pt.toFixed(2)}</td>
      <td class="r num ${sgn(l.impact)}">${l.impact > 0 ? '+' : ''}${l.impact.toFixed(1)}</td></tr>`).join('')}
    </table></div></td></tr>`;
}

function feedRows(feed){
  if (!feed || !feed.length) return '<div class="pos" style="padding:6px 0 10px">No plays involving your players in games currently in progress.</div>';
  const shown = showMinor ? feed : feed.slice(0, 12);
  const rows = shown.map(e => {
    const fors = [...new Set(e.players.flatMap(p => p.for))].join(', ');
    const agst = [...new Set(e.players.flatMap(p => p.against))].join(', ');
    const who = e.players.map(p => `<b>${p.name}</b>${p.role ? ` <span class="pos">(${p.role})</span>` : ''}${p.pts != null ? ` <span class="num pos">${p.pts.toFixed(1)}</span>` : ''}`).join('<br>');
    return `<div class="fev ${e.scoring ? 'score' : ''}">
      <span class="num pos ft">${e.t}<br><span style="font-size:11px">${e.q} ${e.clock}</span></span>
      <span class="fname">${who}<br><span class="pos" style="font-size:12px">${e.game}</span></span>
      <span class="fwhat ${e.scoring ? '' : 'pos'}">${e.text}</span>
      <span class="tags long">${fors}</span>
      <span class="tags short">${agst}</span>
    </div>`;
  }).join('');
  const head = `<div class="fev fhead"><span></span><span>Player</span><span>Play</span><span>Yours in</span><span>Against in</span></div>`;
  const more = feed.length > 12 ? `<div class="pos" style="font-size:12px;margin-top:6px"><a href="#" onclick="showMinor=!showMinor;tick();return false">${showMinor ? 'show fewer' : `show all ${feed.length}`}</a></div>` : '';
  return head + rows + more;
}

function bookRows(book){
  const sgn = x => x > 0 ? 'long' : (x < 0 ? 'short' : 'pos');
  return book.map(p => {
    const tags = [...p.for, ...p.against.map(x => '¬' + x)].join(', ');
    const done = Math.abs(p.impact) < 0.05;
    return `<tr class="${done ? 'done' : ''} prow" onclick="toggle('p:${p.id}')">
      <td>${p.name} <span class="pos">${p.pos} ${p.team}</span></td>
      <td class="r num">${p.pts === null ? '' : p.pts.toFixed(1)}</td>
      <td class="r num pos">${p.proj === null || p.proj === undefined ? '' : p.proj.toFixed(1)}</td>
      <td class="r num ${sgn(p.impact)}">${p.impact > 0 ? '+' : ''}${p.impact.toFixed(1)}</td>
      <td class="r num ${sgn(p.root)}" style="font-size:12.5px">${p.root > 0 ? '+' : ''}${p.root.toFixed(2)}</td>
      <td class="tags">${tags}</td>
    </tr>${open.has('p:' + p.id) ? playerDetail(p) : ''}`;
  }).join('');
}

function leagueTick(d){
  // Leaguemate view: pick a team, see the chop picture from that seat.
  const key = 'team:' + LEAGUE_ID;
  const fromUrl = new URLSearchParams(location.search).get('team');
  let rid = fromUrl || (() => { try { return localStorage.getItem(key); } catch(e){ return null; } })();
  let me = d.teams.find(t => String(t.rid) === String(rid));
  const picker = `<div class="pick"><label>Your team &nbsp;<select id="team" onchange="pickTeam(this.value)">
    <option value="">— choose —</option>
    ${[...d.teams].sort((a,b) => a.team.localeCompare(b.team)).map(t => `<option value="${t.rid}" ${me && t.rid === me.rid ? 'selected' : ''}>${t.team}</option>`).join('')}
  </select></label></div>`;
  let html = picker;
  if (me){
    open.add(me.league_id);              // field + lineups always shown here
    html += chopCard(me);
    const mine = new Set(me.lineup.map(r => r.pid)), theirs = new Set((me.ref_lineup || []).map(r => r.pid));
    const feed = (d.feed || []).map(e => ({...e, players: e.players
        .filter(p => mine.has(p.id) || theirs.has(p.id))
        .map(p => ({...p, for: mine.has(p.id) ? ['you'] : [], against: theirs.has(p.id) ? [me.proj_ref.name] : []}))}))
      .filter(e => e.players.length);
    html += section('feed', 'Feed', feedRows(feed), feed.length);
  } else {
    html += `<div class="pos" style="margin:12px 0 20px">Pick your team to see your margin, survival odds and lineup.</div>`;
  }
  $('#app').innerHTML = html;
  $('#h1').textContent = d.name;
  $('#meta').textContent = `wk ${d.week} · ${d.updated}` + (d.stale ? ' · stale' : '');
}
function pickTeam(rid){
  try { localStorage.setItem('team:' + LEAGUE_ID, rid); } catch(e){}
  history.replaceState(null, '', rid ? `?team=${rid}` : location.pathname);
  tick();
}

async function tick(){
  let d;
  const url = LEAGUE_ID ? `/api/league/${LEAGUE_ID}` : '/api/state';
  try { d = await (await fetch(url)).json(); }
  catch(e){ $('#meta').textContent = 'offline'; return; }
  if (d.error){ $('#app').innerHTML = `<div class="err">${d.error}</div>`; return; }
  if (LEAGUE_ID) return leagueTick(d);

  const pools = d.leagues.filter(l => l.mode === 'pool').sort((a,b) => a.survive_pct - b.survive_pct);
  const h2h   = d.leagues.filter(l => l.mode === 'h2h');
  const manual= d.leagues.filter(l => l.mode === 'manual');
  const errors= d.leagues.filter(l => l.mode === 'error');

  let html = errors.map(l => `<div class="err">${l.name}: ${l.error}</div>`).join('');
  if (pools.length) html += section('pools', 'Guillotine', pools.map(chopCard).join(''));
  if (h2h.length) html += section('h2h', 'Head to head', h2h.map(h2hRow).join(''));
  if (manual.length) html += section('solo', 'Solo', manual.map(manualRow).join(''));
  html += section('feed', 'Feed', feedRows(d.feed), (d.feed || []).length);
  html += section('book', 'Exposure', `<table>
    <tr><th>Player</th><th class="r">Pts</th><th class="r">Proj final</th><th class="r" title="root x remaining projection: swing still on the table">Impact</th><th class="r" title="pp of survival/win per fantasy point, summed over leagues">Root /pt</th><th>Leagues</th></tr>
    ${bookRows(d.book)}</table>`, d.book.length);
  if (manual.length){
    const bad = manual.flatMap(m => m.unmatched);
    if (bad.length) html += `<div class="err">Unmatched names in manual.json: ${bad.join(', ')}. Fix the spelling to fold them into exposure.</div>`;
  }
  $('#app').innerHTML = html;
  $('#meta').textContent = `wk ${d.week} · ${d.updated}` + (d.stale ? ' · stale' : '');
}
tick();
setInterval(tick, 20000);
</script>
"""

if __name__ == "__main__":
    # macOS AirPlay Receiver squats on 5000; PORT=5050 python blotter.py to dodge it.
    app.run(port=int(os.environ.get("PORT", 5000)), debug=False)
