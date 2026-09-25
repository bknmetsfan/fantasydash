#!/usr/bin/env python3
"""
fantasydash - live cross-league fantasy exposure.

Run:  python blotter.py
Then: http://127.0.0.1:5000

Config lives in CONFIG below. Manual (non-API) teams go in manual.json.
ESPN leagues go in espn.json (see espn.example.json; needs espn_s2 + SWID).
"""

import base64
import hashlib
import zlib
import hmac
import json
import math
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
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
    # Guillotine leagues with manual elimination (Sleeper's disable_elimination):
    # name the team being chopped this week once the league decides, and the
    # waiver view treats their players as the incoming pool and drops them
    # from next week's field. league_id -> owner display name. Clear weekly.
    "pending_chop": {},          # e.g. {"1389721132256473088": "kickersvingames"}
    # Default league for the waiver and chart sections.
    "favorite_league": "1400335104223485952",   # Paris in 1795v2
    "shared_leagues": [
        "1400335104223485952",   # Paris in 1795v2
        "1389721132256473088",   # Degenerates
        "1389387283332874240",   # The Red Queen's League
    ],
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


INJ_OUT = {"Out", "IR", "PUP", "Sus", "COV", "DNR"}   # zero the projection


def load_players():
    """The /players/nfl payload is ~5MB. Fetch every 6h (injury designations
    move during the week), cache to disk."""
    if PLAYER_CACHE.exists() and time.time() - PLAYER_CACHE.stat().st_mtime < 6 * 3600:
        cached = json.loads(PLAYER_CACHE.read_text())
        if any("inj" in v for v in list(cached.values())[:50]):
            return cached
    data = get("/players/nfl")
    slim = {
        pid: {
            "name": p.get("full_name") or p.get("last_name") or pid,
            "pos": p.get("position") or "",
            "team": p.get("team") or "FA",
            "espn_id": p.get("espn_id"),
            "inj": p.get("injury_status"),
        }
        for pid, p in data.items()
    }
    PLAYER_CACHE.write_text(json.dumps(slim))
    return slim


def load_projections(season, week, players=None):
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
    # Sleeper keeps projecting players who've been ruled out; don't.
    for pid in list(data):
        if (players or {}).get(pid, {}).get("inj") in INJ_OUT:
            data[pid] = {}
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
    never negative). Each player's draws are seeded by his id, so the same
    player gets the same luck in every scenario: refresh-to-refresh moves
    and what-if comparisons reflect data, not sampling noise.
    """
    out = {}
    for rid, det in teams.items():
        tot = np.full(n, sum(d["act"] for d in det.values()))
        for pid, d in det.items():
            dist = player_dist(d)
            if dist:
                mu, sd = dist
                rng = np.random.default_rng(zlib.crc32(str(pid).encode()))
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
            "inj": p.get("inj"),
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
    current = {ev["id"] for ev in sb.get("events", [])}
    for gid in [g for g in _plays_cache if g not in current]:
        del _plays_cache[gid]            # last week's games
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


# 'J.Gibbs', and ESPN's two-letter disambiguation 'Mi.Wilson' / 'Ma.Wilson'.
_INITIAL_LAST = re.compile(r"\b([A-Z][a-z]?)\.\s?([A-Z][A-Za-z'\-]+)")


_FG_DIST = re.compile(r"(\d+) yard field goal", re.I)


def play_stats(pl, roles, text):
    """
    {pid: {stat: value}} for one ESPN play, from the play type, net yardage
    and each tagged player's role. Yardage bonuses are season-level, so they
    are not priced here.
    """
    ptype = (pl.get("type") or {}).get("text", "") or ""
    yards = pl.get("statYardage") or 0
    td = bool(pl.get("scoringPlay")) and "touchdown" in text.lower()
    low = text.lower()
    out = {}
    for pid, role in roles.items():
        st = {}
        if role == "pass" and ("pass" in low) and "incomplete" not in low:
            if "intercepted" in low or "Interception" in ptype:
                st["pass_int"] = 1
            elif "attempt succeeds" in low:
                st["pass_2pt"] = 1
            else:
                st["pass_yd"] = yards
                if td:
                    st["pass_td"] = 1
        elif role == "rec":
            if "attempt succeeds" in low:
                st["rec_2pt"] = 1
            elif "incomplete" not in low and "intercepted" not in low:
                st["rec"] = 1
                st["rec_yd"] = yards
                if td:
                    st["rec_td"] = 1
        elif role == "rush" or (role == "pass" and "scrambles" in low):
            if "attempt succeeds" in low:
                st["rush_2pt"] = 1
            else:
                st["rush_yd"] = yards
                if td:
                    st["rush_td"] = 1
        elif role == "kick":
            if "field goal" in low:
                m = _FG_DIST.search(text)
                dist = int(m.group(1)) if m else 0
                if "no good" in low or "missed" in low or "blocked" in low:
                    st["fgmiss"] = 1
                else:
                    bucket = ("fgm_0_19" if dist < 20 else "fgm_20_29" if dist < 30 else "fgm_30_39" if dist < 40
                              else "fgm_40_49" if dist < 50 else "fgm_50_59" if dist < 60 else "fgm_60p")
                    st[bucket] = 1
            elif "extra point is good" in low:
                st["xpm"] = 1
        if role == "fum" or (pid in roles and "fumble recovery (opponent)" in ptype.lower() and role in ("rush", "rec", "pass")):
            if "fumbles" in low and "opponent" in ptype.lower():
                st["fum_lost"] = 1
        out[pid] = st
    return out


def price(stats, scoring, key=None):
    """Points for a stat line under a scoring table. `key` selects Sleeper's
    precomputed pts_* style for ppr/half/std manual teams."""
    if key:
        base = {"pts_ppr": {"rec": 1}, "pts_half_ppr": {"rec": 0.5}, "pts_std": {}}[key]
        scoring = {"pass_yd": .04, "pass_td": 4, "pass_int": -1, "rush_yd": .1, "rush_td": 6,
                   "rec_yd": .1, "rec_td": 6, "fum_lost": -2, "pass_2pt": 2, "rush_2pt": 2, "rec_2pt": 2, **base}
    return round(sum(v * (scoring or {}).get(k, 0) for k, v in stats.items()), 2)


def build_feed(book, players=None, scoring_by_tag=None):
    """
    Plays involving anyone in `book` (dicts with id/name/team/for/against/
    pts), newest first. Players are matched
    from ESPN's participant list (full names), falling back to the
    'J.Gibbs' tokens in the play text matched by initial + surname + team.
    """
    by_name, by_last = {}, {}
    for p in book:
        by_name[norm(p["name"])] = p
        parts = p["name"].split()
        if parts:
            by_last.setdefault((norm(parts[-1]), p["team"]), []).append(p)

    def by_prefix(ini, last, team):
        cands = [c for c in by_last.get((norm(last), team), []) if c["name"].lower().startswith(ini.lower())]
        return cands[0] if len(cands) == 1 else None
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
                for mt in _INITIAL_LAST.finditer(text):
                    ini, last = mt.groups()
                    for t in game_teams:
                        hit = by_prefix(ini, last, t)
                        if not hit:
                            continue
                        hits[hit["id"]] = hit
                        # Role from the words around the name, not its position:
                        # pre-snap notes ("A.Belton reported in as eligible.") come first.
                        before = text[max(0, mt.start() - 14):mt.start()].lower()
                        after = text[mt.end():mt.end() + 24].lower()
                        if after.startswith(" pass") or after.startswith(" scrambles") or after.startswith(" sacked"):
                            role = "rush" if after.startswith(" scrambles") else "pass"
                        elif before.endswith("to ") or before.endswith("for ") or " to " in before[-4:]:
                            role = "rec"
                        elif after.startswith(" kicks") or "field goal" in after or "extra point" in after:
                            role = "kick"
                        elif after.startswith(" fumbles"):
                            role = "fum"
                        else:
                            role = "rush"
                        roles.setdefault(hit["id"], role)
            if not hits:
                continue
            try:
                ts = time.mktime(time.strptime(pl.get("wallclock", "")[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
                if time.localtime(ts).tm_isdst:
                    ts += 3600
            except (ValueError, TypeError):
                ts = time.time()
            q = pl.get("period", {}).get("number")
            stats_by_pid = play_stats(pl, roles, text)
            def deltas(h):
                d, mults = {}, {}
                for tag in list(h["for"]) + list(h["against"]):
                    sc = (scoring_by_tag or {}).get(tag.replace(" (bench)", ""))
                    if sc is not None:
                        base = price(stats_by_pid.get(h["id"], {}), sc["scoring"], sc.get("key"))
                        m = (sc.get("mults") or {}).get(h["id"], 1)
                        d[tag] = round(base * m, 2)
                        if m != 1:
                            mults[tag] = {"base": base, "mult": m}
                return d, mults
            out.append({
                "id": pl.get("id"), "ts": ts, "t": time.strftime("%H:%M", time.localtime(ts)),
                "game": g["label"], "q": f"Q{q}" if q and q <= 4 else "OT",
                "clock": (pl.get("clock") or {}).get("displayValue", ""),
                "text": re.sub(r"^\((?:Shotgun|No Huddle|No Huddle, Shotgun)\)\s*", "", text),
                "scoring": bool(pl.get("scoringPlay")),
                "players": [{"id": h["id"], "name": h["name"], "for": h["for"], "against": h["against"],
                             "pts": h["pts"], "role": roles.get(h["id"], ""),
                             "deltas": deltas(h)[0], "mults": deltas(h)[1]}
                            for h in hits.values()],
            })
    out.sort(key=lambda e: -e["ts"])
    return out[:FEED_MAX]


def win_pct(a, b):
    return round(100 * float(np.mean(a > b) + 0.5 * np.mean(a == b)), 1)


def my_rid_missing(mine, by_roster):
    """True once we've been chopped (or otherwise have no live matchup)."""
    return mine["roster_id"] not in by_roster


def best_lineup_slots(player_ids, slots, value, players):
    """best_lineup, but returns [(slot, pid)] in roster_positions order."""
    avail = {p: value(p) for p in player_ids if p and p != "0"}
    named = [(i, sl) for i, sl in enumerate(slots) if sl not in NON_SLOTS]
    order = sorted(named, key=lambda t: len(ELIGIBLE.get(t[1], {t[1]})))
    picked = {}
    for i, slot in order:
        elig = ELIGIBLE.get(slot, {slot})
        cands = [p for p in avail if players.get(p, {}).get("pos") in elig]
        if cands:
            best = max(cands, key=avail.get)
            picked[i] = (slot, best)
            del avail[best]
    return [picked[i] for i, _ in named if i in picked]


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


def load_manual(players, projections, stats, games, week):
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
    # Prefer rostered players: name clashes with free agents (Kenneth Walker
    # the WR, Antonio Williams the RB) would otherwise win on dict order.
    index = {}
    for pid, p in sorted(players.items(), key=lambda kv: kv[1].get("team") in (None, "", "FA")):
        index.setdefault(norm(p["name"]), pid)

    out = []
    for team in json.loads(MANUAL_FILE.read_text()):
        scoring = team.get("scoring", "ppr")
        stale = team.get("week") is not None and team.get("week") != week
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
            "stale_week": team.get("week") if stale else None,
            "scoring": ELIMINATOR if scoring == "eliminator" else None,
            "scoring_key": None if scoring == "eliminator" else "pts_" + scoring,
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
ESPN_SCORING_MAP = {
    "PY": ["pass_yd"], "PTD": ["pass_td"], "INTT": ["pass_int"], "2PC": ["pass_2pt"],
    "RY": ["rush_yd"], "RTD": ["rush_td"], "2PR": ["rush_2pt"],
    "REC": ["rec"], "REY": ["rec_yd"], "RETD": ["rec_td"], "2PRE": ["rec_2pt"],
    "FUML": ["fum_lost"], "PAT": ["xpm"], "FGM": ["fgmiss"],
    "FG0": ["fgm_0_19", "fgm_20_29", "fgm_30_39"], "FG40": ["fgm_40_49"], "FG50": ["fgm_50_59"], "FG60": ["fgm_60p"],
    "PRTD": ["pr_td"], "KRTD": ["kr_td"],
}


def espn_scoring(fmt):
    """espn-api settings.scoring_format -> {sleeper stat: points}."""
    out = {}
    for row in fmt or []:
        for k in ESPN_SCORING_MAP.get(row.get("abbr"), []):
            out[k] = row.get("points") or 0.0
    return out


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

    def bench_rows(lineup):
        det = {}
        for bp in lineup:
            if bp.slot_position not in ("BE", "IR"):
                continue
            pid = resolve(bp)
            team = pid if pid in games else players.get(pid, extras.get(pid, {})).get("team")
            det[pid] = {"act": bp.points or 0.0, "proj": bp.projected_points or 0.0,
                        "rem": games.get(team, 1.0), "pos": "DEF" if bp.position == "D/ST" else bp.position}
        det = dict(sorted(det.items(), key=lambda kv: -(kv[1]["act"] + kv[1]["proj"] * kv[1]["rem"])))
        return lineup_rows(det, {**players, **extras}, stats)

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
        def fetch():
            lg = League(league_id=int(lc["league_id"]), year=int(CONFIG["season"]),
                        espn_s2=cfg.get("espn_s2"), swid=cfg.get("swid"))
            return lg, lg.box_scores(week)
        try:
            # espn-api issues requests with no timeout; when ESPN hangs, so
            # would the whole build (and every thread queued behind it).
            ex = ThreadPoolExecutor(max_workers=1)
            try:
                lg, boxes = ex.submit(fetch).result(timeout=20)
            finally:
                ex.shutdown(wait=False)        # a hung fetch thread is abandoned, not joined
        except FutureTimeout:
            out.append({"league_id": lid, "name": lc.get("name") or f"ESPN {lc['league_id']}",
                        "mode": "error", "error": "ESPN API not responding (timed out after 20s); retrying next refresh"})
            continue
        except Exception as exc:
            msg = str(exc)
            if "credentials" in msg.lower() or "401" in msg or "403" in msg:
                msg += " — the espn_s2 cookie has probably expired: re-copy it from a logged-in espn.com session and run `fly secrets set ESPN_S2=...`"
            out.append({"league_id": lid, "name": lc.get("name") or f"ESPN {lc['league_id']}",
                        "mode": "error", "error": msg})
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
                    "scoring": espn_scoring(getattr(lg.settings, "scoring_format", None)),
                    "my_points": round(getattr(box, f"{me}_score") or 0.0, 2),
                    "my_proj": round(sum(my_proj.values()), 2),
                    "my_to_play": to_play(my_det),
                    "lineup": lineup_rows(my_det, {**players, **extras}, stats),
                    "bench_lineup": bench_rows(getattr(box, f"{me}_lineup")),
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

LOG_IDLE_SECONDS = 3600    # snapshot cadence when no game is in progress


def log_snapshot(season, week, leagues):
    """Calibration snapshot: every player's act/proj/rem, per league. Every
    log_seconds while a game is in progress, hourly otherwise — logging
    around the clock filled the volume (~55 MB/day) with identical rows."""
    if not CONFIG["log_seconds"] or time.time() - _log_last["ts"] < CONFIG["log_seconds"]:
        return
    live = any(0 < r[-1] < 1 for lg in leagues for r in lg.get("_log", []))
    if not live and time.time() - _log_last["ts"] < LOG_IDLE_SECONDS:
        for lg in leagues:
            lg.pop("_log", None)
        return
    ts = int(time.time())
    con = connect_log()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            ts INTEGER, season TEXT, week INTEGER, league_id TEXT, league TEXT,
            rid TEXT, mine INTEGER, pid TEXT, name TEXT, pos TEXT, team TEXT,
            act REAL, proj REAL, rem REAL);
        CREATE INDEX IF NOT EXISTS players_wk ON players (season, week, pid);
        CREATE INDEX IF NOT EXISTS players_lg ON players (season, week, league_id, ts);
        CREATE TABLE IF NOT EXISTS teams (
            season TEXT, week INTEGER, league_id TEXT, rid TEXT, name TEXT,
            PRIMARY KEY (season, week, league_id, rid));
        CREATE TABLE IF NOT EXISTS leagues (
            ts INTEGER, season TEXT, week INTEGER, league_id TEXT, league TEXT, mode TEXT,
            my_points REAL, my_proj REAL, pct REAL, sens REAL, rank INTEGER, field_size INTEGER,
            to_play INTEGER, live INTEGER);
    """ + TEAM_SNAPS_DDL)
    prow, lrow, trow = [], [], []
    for lg in leagues:
        if lg["mode"] == "error":
            continue
        for r in lg.pop("_log", []):
            prow.append((ts, season, week, lg["league_id"], lg["name"], *r))
        for f in lg.get("field") or []:
            trow.append((season, week, lg["league_id"], str(f["rid"]), f["name"]))
        for pair in lg.get("matchups") or []:
            for t in pair:
                trow.append((season, week, lg["league_id"], str(t["rid"]), t["name"]))
        tp = lg.get("my_to_play") or (None, None)
        lrow.append((ts, season, week, lg["league_id"], lg["name"], lg["mode"],
                     lg.get("my_points"), lg.get("my_proj"),
                     lg.get("survive_pct", lg.get("win_pct")), lg.get("sens"),
                     lg.get("rank"), lg.get("field_size"), tp[0], tp[1]))
    con.executemany("INSERT INTO players VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", prow)
    con.executemany("INSERT OR REPLACE INTO team_snaps VALUES (?,?,?,?,?,?,?)", team_rows(prow))
    con.executemany("INSERT INTO leagues VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", lrow)
    con.executemany("INSERT OR REPLACE INTO teams VALUES (?,?,?,?,?)", trow)
    con.commit()
    con.close()
    _log_last["ts"] = time.time()


# Per-team totals per snapshot, so the chart never sums the players table.
TEAM_SNAPS_DDL = """
        CREATE TABLE IF NOT EXISTS team_snaps (
            season TEXT, week INTEGER, league_id TEXT, ts INTEGER, rid TEXT, pts REAL, proj REAL,
            PRIMARY KEY (season, league_id, week, ts, rid));
        CREATE INDEX IF NOT EXISTS leagues_lg ON leagues (league_id, week);
"""


def team_rows(prow):
    """players rows (ts, season, week, lid, league, rid, mine, pid, name, pos,
    team, act, proj, rem) -> team_snaps rows with live pts and blended final."""
    agg = {}
    for ts, season, week, lid, _, rid, _, _, _, _, _, act, proj, rem in prow:
        a = agg.setdefault((season, week, lid, ts, rid), [0.0, 0.0])
        a[0] += act
        a[1] += act + proj * rem
    return [(se, wk, lid, ts, rid, round(p, 2), round(f, 2)) for (se, wk, lid, ts, rid), (p, f) in agg.items()]


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
            "field_size": n,                # field is sent once, see api_league
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
                                         "pts": r["act"], "for": ["_"], "against": []})
    feed = build_feed(list(roster.values()), None, {"_": {"scoring": entry.get("scoring")}})
    for e in feed:                      # the "_" tag was only for pricing
        for pp in e["players"]:
            pp["for"] = []
    return {"teams": teams, "feed": feed, "field": field}


# ----------------------------------------------------------------------------
# Build the picture
# ----------------------------------------------------------------------------

def build():
    players = load_players()
    state = get("/state/nfl")
    week = state.get("week") or 1
    projections = load_projections(CONFIG["season"], week, players)
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
        matchups = get(f"/league/{lid}/matchups/{week}") or []
        # Eliminated guillotine rosters linger in the matchup list with no
        # players and no lineup; they'd otherwise sit at the bottom of the
        # field forever. Also covers the Tue/Wed gap before Sleeper builds
        # the new week (no matchups at all -> skip the league this refresh).
        roster_players = {r["roster_id"]: r.get("players") or [] for r in rosters}
        matchups = [m for m in matchups
                    if roster_players.get(m["roster_id"]) or any(p and p != "0" for p in (m.get("starters") or []))]
        if not matchups:
            continue
        by_roster = {m["roster_id"]: m for m in matchups}

        mine = next((r for r in rosters if r.get("owner_id") == uid), None)
        if mine is None or my_rid_missing(mine, by_roster):
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
            ids = [p for p in (m.get("starters") or []) if p and p != "0"]
            if best_ball or not ids:
                # Best ball, or a lineup nobody has set yet: assume the optimal one.
                ids = best_lineup(m.get("players") or [], lg.get("roster_positions") or [], val, players)
            det = {}
            for p in ids:
                team = p if p in games else players.get(p, {}).get("team")
                det[p] = {"act": actual.get(p) or 0.0, "proj": proj_pts(p, projections, scoring) or 0.0,
                          "rem": games.get(team, 1.0), "pos": players.get(p, {}).get("pos", "")}
            return ids, {p: val(p) for p in ids}, det

        def bench_det(m, lineup_ids):
            actual = m.get("players_points") or {}
            det = {}
            for p in m.get("players") or []:
                if p in lineup_ids or not p or p == "0":
                    continue
                team = p if p in games else players.get(p, {}).get("team")
                det[p] = {"act": actual.get(p) or 0.0, "proj": proj_pts(p, projections, scoring) or 0.0,
                          "rem": games.get(team, 1.0), "pos": players.get(p, {}).get("pos", "")}
            # Highest projected first so the "should have started him" case is on top.
            return dict(sorted(det.items(), key=lambda kv: -(kv[1]["act"] + kv[1]["proj"] * kv[1]["rem"])))

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
            "scoring": scoring,
            "my_proj": round(sum(my_proj.values()), 2),
            "my_to_play": to_play(my_det),
            "lineup": lineup_rows(my_det, players, stats),
            "_log": [r for rid, (_, _, det) in proj_by_rid.items()
                     for r in log_rows(rid, rid == my_rid, det, players)],
            # Best ball: the rest of the roster can still play its way into
            # the lineup, so the feed watches them too (tagged bench).
            "bench": [{"id": p, "pts": round((my_m.get("players_points") or {}).get(p) or 0.0, 2)}
                      for p in (my_m.get("players") or []) if best_ball and p not in my_lineup],
            "bench_lineup": lineup_rows(bench_det(my_m, my_lineup), players, stats),
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
            for f in field:
                f["lineup"] = lineup_rows(proj_by_rid[f["rid"]][2], players, stats)
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
    out_leagues.extend(load_manual(players, projections, stats, games, week))

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
    scoring_by_tag = {lg["name"][:14]: {"scoring": lg.get("scoring"), "key": lg.get("scoring_key"),
                                        "mults": lg.get("mults") or {}}
                      for lg in out_leagues if lg["mode"] != "error"}
    feed = build_feed(list(watch.values()), players, scoring_by_tag)
    log_snapshot(CONFIG["season"], week, out_leagues)   # pops the _log rows
    for lg in out_leagues:
        lg.pop("_log", None)

    return {"week": week, "leagues": out_leagues, "book": book, "feed": feed,
            "favorite": CONFIG["favorite_league"], "updated": time.strftime("%H:%M:%S")}


AUTH_COOKIE = "fd_auth"
AUTH_DAYS = 30


def auth_token(pw):
    """Cookie value proving the password was entered: HMAC over a fixed
    message keyed by the password, so changing the password revokes it."""
    return hmac.new(hashlib.sha256(pw.encode()).digest(), b"fantasydash-session", hashlib.sha256).hexdigest()


@app.before_request
def basic_auth():
    """
    Single shared password via BLOTTER_PASSWORD; open when unset (local).
    A successful Basic login also sets a 30-day cookie, so browsers that
    don't reattach Basic credentials to fetch() (iOS Safari) only prompt
    once per device.
    """
    pw = os.environ.get("BLOTTER_PASSWORD")
    if not pw or request.path in ("/healthz", "/tick") or request.path.startswith(("/l/", "/api/league/")):
        return None
    if request.path.startswith("/api/history/") and request.path.rsplit("/", 1)[-1] in CONFIG["shared_leagues"]:
        return None
    if hmac.compare_digest(request.cookies.get(AUTH_COOKIE, ""), auth_token(pw)):
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
    request.set_auth_cookie = True


@app.after_request
def set_auth_cookie(resp):
    if getattr(request, "set_auth_cookie", False):
        resp.set_cookie(AUTH_COOKIE, auth_token(os.environ["BLOTTER_PASSWORD"]), max_age=AUTH_DAYS * 86400,
                        httponly=True, secure=request.is_secure, samesite="Lax")
    return resp


def current_state():
    """
    Cached build, refreshed at most every poll_seconds. Returns (data, error).
    If a build is already running, serve what we have rather than queueing:
    a slow upstream must not tie up every request thread.
    """
    now = time.time()
    fresh = _cache["data"] is not None and now - _cache["ts"] <= CONFIG["poll_seconds"]
    if fresh:
        return _cache["data"], None
    have = _cache["data"] is not None
    got = _build_lock.acquire(timeout=60) if not have else _build_lock.acquire(blocking=False)
    if not got:
        return (_cache["data"], None) if have else (None, "build in progress")
    try:
        if _cache["data"] is None or now - _cache["ts"] > CONFIG["poll_seconds"]:
            try:
                _cache["data"] = build()
                _cache["ts"] = time.time()
            except Exception as exc:
                if _cache["data"] is None:
                    return None, str(exc)
                _cache["data"]["stale"] = str(exc)
        return _cache["data"], None
    finally:
        _build_lock.release()


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
                    "name": lg["name"], "teams": lg["shared"]["teams"], "feed": lg["shared"]["feed"],
                    "field": lg["shared"]["field"]})


_roster_names = {}
_waiver_cache = {}        # league_id -> (ts, report)
WAIVER_POS = {"QB", "RB", "WR", "TE", "K", "DEF"}
WAIVER_TOP = 12           # rows shown
WAIVER_LOG = 30           # free agents snapshotted for the bids database
_bids_last = {}           # league_id -> ts of last transactions sync


# Endgame = still clearly a starter the rest of the way: top-N at the
# position by rest-of-season value. Source, in order of preference:
#   1. a FantasyPros ROS export dropped in rankings/ (see fp_ros_rank)
#   2. Sleeper's weekly projections summed over the remaining weeks.
# Not Sleeper's season row: it is a full-season total (18 games for everyone,
# weeks already played included, injuries ignored) that doesn't move.
ENDGAME_RANK = {"QB": 12, "RB": 10, "WR": 15, "TE": 5}
RANKINGS_DIR = HERE / "rankings"
LAST_WEEK = 18
_season_cache = {"ts": 0.0, "rank": {}, "source": ""}


def sleeper_ros_rank(players):
    """({pid: positional rank}, source) from Sleeper's weekly pts_ppr
    projections summed from the current week through LAST_WEEK."""
    week = get("/state/nfl").get("week") or 1
    qs = "&".join(f"position[]={p}" for p in ENDGAME_RANK)
    weeks = range(week, LAST_WEEK + 1)
    with ThreadPoolExecutor(8) as ex:
        tables = list(ex.map(lambda w: get(f"/{CONFIG['season']}/{w}?season_type=regular&{qs}", base=PROJ_BASE), weeks))
    tot = {}
    for rows in tables:
        for row in rows:
            tot[row["player_id"]] = tot.get(row["player_id"], 0.0) + ((row.get("stats") or {}).get("pts_ppr") or 0.0)
    by_pos = {}
    for pid, pts in tot.items():
        pos = players.get(pid, {}).get("pos")
        if pos in ENDGAME_RANK:
            by_pos.setdefault(pos, []).append((pid, pts))
    rank = {}
    for lst in by_pos.values():
        rank.update({pid: i + 1 for i, (pid, _) in enumerate(sorted(lst, key=lambda t: -t[1]))})
    return rank, f"Sleeper ROS wk{week}–{LAST_WEEK}"


def fp_ros_rank(players):
    """({pid: positional rank}, source) from FantasyPros CSV export(s) in
    rankings/, or ({}, "") when there are none. Tolerant of both export
    shapes: rankings (POS like "WR12" carries the positional rank) and
    projections (FPTS column, ranked within position). Several files are
    merged, so per-position projection exports work too."""
    import csv
    files = sorted(RANKINGS_DIR.glob("*.csv")) if RANKINGS_DIR.exists() else []
    if not files:
        return {}, ""
    # Names -> candidate pids; FantasyPros team codes differ in places.
    alias = {"JAC": "JAX", "LA": "LAR", "WSH": "WAS"}
    index = {}
    for pid, p in players.items():
        if p.get("pos") in ENDGAME_RANK:
            index.setdefault(norm(p.get("name", "")), []).append(pid)
    by_pos, matched, seen = {}, 0, 0
    for f in files:
        with f.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
        # The header is the first row naming a player column (exports can
        # carry a title line above it).
        hi = next((i for i, r in enumerate(rows) if any(c.strip().upper() in ("PLAYER NAME", "PLAYER", "NAME") for c in r)), None)
        if hi is None:
            continue
        head = [c.strip().upper() for c in rows[hi]]
        col = lambda *names: next((head.index(n) for n in names if n in head), None)
        c_name, c_pos, c_team = col("PLAYER NAME", "PLAYER", "NAME"), col("POS", "POSITION"), col("TEAM")
        c_pts, c_rk = col("FPTS", "FPTS/G", "PTS"), col("RK", "RANK", "ECR")
        for order, r in enumerate(rows[hi + 1:]):
            if len(r) <= c_name or not r[c_name].strip():
                continue
            name = re.sub(r"\s*\([A-Z]{2,3}\s*-\s*[A-Z]+\)\s*$", "", r[c_name]).strip()   # "Name (TEAM - POS)" form
            team = r[c_team].strip().upper() if c_team is not None and c_team < len(r) else ""
            team = alias.get(team, team)
            pos_raw = r[c_pos].strip().upper() if c_pos is not None and c_pos < len(r) else ""
            m = re.match(r"([A-Z]+)(\d+)?", pos_raw)
            seen += 1
            cands = index.get(norm(name), [])
            if m:
                cands = [p for p in cands if players[p].get("pos") == m.group(1)] or cands
            if team and len(cands) > 1:
                cands = [p for p in cands if players[p].get("team") == team] or cands
            cands = sorted(cands, key=lambda p: players[p].get("team") in (None, "", "FA"))
            if not cands:
                continue
            pid = cands[0]
            pos = players[pid].get("pos")
            if pos not in ENDGAME_RANK:
                continue
            matched += 1
            if m and m.group(2):
                key = int(m.group(2))                       # explicit positional rank
            elif c_pts is not None and c_pts < len(r):
                try:
                    key = -float(r[c_pts].replace(",", ""))  # rank by points, high first
                except ValueError:
                    key = order
            else:
                key = float(r[c_rk]) if c_rk is not None and c_rk < len(r) and r[c_rk].strip() else order
            by_pos.setdefault(pos, {})[pid] = key
    rank = {}
    for d in by_pos.values():
        rank.update({pid: i + 1 for i, pid in enumerate(sorted(d, key=d.get))})
    when = time.strftime("%m-%d", time.localtime(max(f.stat().st_mtime for f in files)))
    return rank, f"FantasyPros ROS {when} ({matched}/{seen} matched)"


def season_pos_rank(players):
    """{pid: rest-of-season rank at position}; FantasyPros file if present,
    else Sleeper's remaining weekly projections. Cached 6h; a failed fetch
    keeps the previous ranks."""
    if time.time() - _season_cache["ts"] < 6 * 3600 and _season_cache["rank"]:
        return _season_cache["rank"]
    try:
        rank, source = fp_ros_rank(players)
        if not rank:
            rank, source = sleeper_ros_rank(players)
    except Exception as e:
        print("season_pos_rank:", e)
        return _season_cache["rank"]
    if rank:
        _season_cache.update(ts=time.time(), rank=rank, source=source)
    return _season_cache["rank"]


def tier_of(frac, pos=None, season_rank=None):
    """1 endgame: top-N at position for the season. 2 starter: would start
    for a third of the league this week. 3 fill-in."""
    if pos in ENDGAME_RANK and season_rank is not None and season_rank <= ENDGAME_RANK[pos]:
        return 1
    return 2 if frac >= 0.33 else 3


def waiver_report(lid, as_rid=None):
    """
    Next-week view of a guillotine league, pregame: everyone on their
    optimal lineup by projection. Positional holes (my starters vs the
    field) and the marginal value of the top free agents: projected-final
    delta, chop % before/after, who they'd displace.
    """
    c = _waiver_cache.get((lid, as_rid))
    if c and time.time() - c[0] < 600:
        return c[1]
    players = load_players()
    week = (get("/state/nfl").get("week") or 1)
    projections = load_projections(CONFIG["season"], week, players)
    lg = get(f"/league/{lid}")
    scoring = lg.get("scoring_settings") or {}
    slots = lg.get("roster_positions") or []
    users = {u["user_id"]: (u.get("display_name") or "?") for u in get(f"/league/{lid}/users")}
    rosters = [r for r in get(f"/league/{lid}/rosters") if r.get("players")]
    uid = get(f"/user/{CONFIG['sleeper_username']}")["user_id"]
    me_roster = next((r for r in rosters if r.get("owner_id") == uid), None)
    mine = next((r for r in rosters if str(r["roster_id"]) == str(as_rid)), None) if as_rid else me_roster
    if mine is None:
        return {"error": "no live roster in this league"}
    my_rid = mine["roster_id"]
    val = lambda p: proj_pts(p, projections, scoring) or 0.0

    # Pending chop (manual-elimination leagues only, from CONFIG): that
    # roster's players are the incoming pool and it leaves next week's field.
    # Leagues where Sleeper eliminates automatically need nothing: the roster
    # is already emptied and its players are plain free agents.
    chop_pool, chop_name = set(), None
    want = CONFIG["pending_chop"].get(lid)
    chopped = next((r for r in rosters if want and users.get(r.get("owner_id")) == want), None)
    if chopped and chopped["roster_id"] != my_rid:
        chop_pool = set(chopped["players"])
        chop_name = want
        rosters = [r for r in rosters if r is not chopped]

    def det_for(ids):
        lineup = best_lineup(ids, slots, val, players)
        return {p: {"act": 0.0, "proj": val(p), "rem": 1.0, "pos": players.get(p, {}).get("pos", "")} for p in lineup}

    def slot_labels(assigned):
        """[(slot, pid)] -> [('RB1', pid), ('RB2', pid), ('FLEX', pid), ...]"""
        counts = {}
        for sl, _ in assigned:
            counts[sl] = counts.get(sl, 0) + 1
        seen, out = {}, []
        for sl, pid in assigned:
            seen[sl] = seen.get(sl, 0) + 1
            label = sl.replace("SUPER_FLEX", "SF")
            out.append((f"{label}{seen[sl]}" if counts[sl] > 1 else label, pid))
        return out

    by_slot = {r["roster_id"]: slot_labels(best_lineup_slots(r["players"], slots, val, players)) for r in rosters}
    slot_order = [lbl for lbl, _ in next(iter(by_slot.values()), [])]

    dets = {r["roster_id"]: det_for(r["players"]) for r in rosters}
    sims = simulate(dets, CONFIG["sims"])
    rids = [r["roster_id"] for r in rosters]

    def chop_pct(my_sims):
        mat = np.array([my_sims if rid == my_rid else sims[rid] for rid in rids])
        return round(100 * float(np.mean(mat.argmin(axis=0) == rids.index(my_rid))), 1)

    my_det = dets[my_rid]
    base = {"proj": round(sum(d["proj"] for d in my_det.values()), 2), "chop_pct": chop_pct(sims[my_rid])}

    # The whole field on optimal lineups: total, chop %, and per-position split.
    mat_all = np.array([sims[rid] for rid in rids])
    chop_all = np.bincount(mat_all.argmin(axis=0), minlength=len(rids)) / mat_all.shape[1]
    field = []
    budget = (lg.get("settings") or {}).get("waiver_budget") or 0
    for rid, r in zip(rids, rosters):
        det = dets[rid]
        split = {lbl: round(val(pid), 2) for lbl, pid in by_slot[rid]}
        field.append({"rid": rid, "name": users.get(r.get("owner_id"), f"Roster {rid}"),
                      "faab": budget - ((r.get("settings") or {}).get("waiver_budget_used") or 0),
                      "proj": round(sum(d["proj"] for d in det.values()), 2),
                      "chop_pct": round(100 * float(chop_all[rids.index(rid)]), 1),
                      "split": split, "me": rid == my_rid})
    field.sort(key=lambda f: -f["proj"])

    # Holes by lineup slot: my starter in each slot vs everyone's starter in
    # that slot (RB1 vs the field's RB1s, FLEX vs FLEXes, ...).
    holes = []
    for lbl in slot_order:
        vals = {rid: val(pid) for rid, pairs in by_slot.items() for l2, pid in pairs if l2 == lbl}
        if my_rid not in vals:
            continue
        me = vals[my_rid]
        my_pid = next(pid for l2, pid in by_slot[my_rid] if l2 == lbl)
        ranked = sorted(vals.values(), reverse=True)
        holes.append({"pos": lbl, "who": players.get(my_pid, {}).get("name", my_pid),
                      "mine": round(me, 2), "rank": ranked.index(me) + 1,
                      "of": len(ranked), "median": round(float(np.median(ranked)), 2),
                      "best": round(ranked[0], 2), "gap": round(me - float(np.median(ranked)), 2)})

    # Free agents: projected players nobody in the (surviving) league rosters,
    # plus the pending chop pool identified above.
    rostered = {p for r in rosters for p in r["players"]}
    # Only positions the league has a slot for (no DEF/K in leagues without them).
    usable = set().union(*(ELIGIBLE.get(sl, {sl}) for sl in slots if sl not in NON_SLOTS)) & WAIVER_POS
    pool = [pid for pid in set(projections) | chop_pool if pid not in rostered
            and players.get(pid, {}).get("pos") in usable and players.get(pid, {}).get("team", "FA") != "FA"]
    # Screen the whole pool cheaply first (lineup delta, no sims): a gem with a
    # modest projection can be worth more to this roster than the top name on
    # the wire. Keep the top by projection too, so the bids snapshot still
    # covers the players rivals will actually bid on.
    base_proj_mine = sum(d["proj"] for d in dets[my_rid].values())
    screen = {pid: sum(val(p) for p in best_lineup(list(mine["players"]) + [pid], slots, val, players))
                   - base_proj_mine for pid in pool}
    by_val = sorted(pool, key=lambda p: -val(p))[:WAIVER_LOG]
    by_delta = sorted(pool, key=lambda p: (-screen[p], -val(p)))[:WAIVER_LOG]
    fas = sorted(set(by_val) | set(by_delta), key=lambda p: (-screen[p], -val(p)))
    # Positional rank among everyone (rostered + FA): weekly for the board,
    # season-long for the endgame tier.
    pos_rank = {}
    for pos in usable:
        ranked = sorted((pid for pid in projections if players.get(pid, {}).get("pos") == pos), key=lambda p: -val(p))
        pos_rank.update({pid: i + 1 for i, pid in enumerate(ranked)})
    srank = season_pos_rank(players)
    old_lineup = set(my_det)
    base_proj = {rid: sum(d["proj"] for d in det.values()) for rid, det in dets.items()}
    base_chop = {rid: 100 * float(chop_all[rids.index(rid)]) for rid in rids}

    def chop_pct_for(rid, team_sims):
        mat = np.array([team_sims if r == rid else sims[r] for r in rids])
        return 100 * float(np.mean(mat.argmin(axis=0) == rids.index(rid)))

    cands = []
    for pid in fas:
        det2 = det_for(list(mine["players"]) + [pid])
        s2 = simulate({my_rid: det2}, CONFIG["sims"])[my_rid]
        proj2 = round(sum(d["proj"] for d in det2.values()), 2)
        displaced = [players.get(p, {}).get("name", p) for p in old_lineup - set(det2)]
        p = players.get(pid, {})
        # Demand: the same add from every other seat. Who would start him,
        # and who gains the most chop safety from him.
        wants = []
        for r in rosters:
            rid = r["roster_id"]
            if rid == my_rid:
                continue
            d_other = det_for(list(r["players"]) + [pid])
            if pid not in d_other:
                continue
            gain = sum(d["proj"] for d in d_other.values()) - base_proj[rid]
            swing = chop_pct_for(rid, simulate({rid: d_other}, CONFIG["sims"])[rid]) - base_chop[rid]
            wants.append({"name": users.get(r.get("owner_id"), f"Roster {rid}"), "delta": round(gain, 2),
                          "chop_swing": round(swing, 1), "faab": budget - ((r.get("settings") or {}).get("waiver_budget_used") or 0)})
        wants.sort(key=lambda w: w["chop_swing"])
        frac = len(wants) / max(1, len(rosters) - 1)
        cands.append({"id": pid, "name": p.get("name", pid), "pos": p.get("pos", ""), "team": p.get("team", ""),
                      "proj": round(val(pid), 2), "starts": pid in det2, "chop_pool": pid in chop_pool,
                      "pos_rank": pos_rank.get(pid), "season_rank": srank.get(pid),
                      "tier": tier_of(frac, p.get("pos"), srank.get(pid)),
                      "max_delta": round(max((w["delta"] for w in wants), default=0.0), 2),
                      "max_swing": round(min((w["chop_swing"] for w in wants), default=0.0), 1),
                      "proj_after": proj2, "delta": round(proj2 - base["proj"], 2),
                      "chop_after": chop_pct(s2), "displaces": displaced,
                      "demand": {"starts": len(wants), "of": len(rosters) - 1, "top": wants[:3]}})
    # My lineup by slot, what's actually set on Sleeper, and the best bench
    # alternatives per slot so close calls are visible.
    assigned = best_lineup_slots(mine["players"], slots, val, players)
    optimal_ids = [pid for _, pid in assigned]
    bench = [p for p in mine["players"] if p not in optimal_ids]
    set_now = [p for p in (mine.get("starters") or []) if p and p != "0"]
    lineup_rows_out = []
    for slot, pid in assigned:
        elig = ELIGIBLE.get(slot, {slot})
        alts = sorted((b for b in bench if players.get(b, {}).get("pos") in elig), key=lambda b: -val(b))[:2]
        lineup_rows_out.append({
            "slot": slot, "id": pid, "name": players.get(pid, {}).get("name", pid),
            "pos": players.get(pid, {}).get("pos", ""), "team": players.get(pid, {}).get("team", ""),
            "inj": players.get(pid, {}).get("inj"),
            "proj": round(val(pid), 2), "set": pid in set_now,
            "alts": [{"name": players.get(b, {}).get("name", b), "pos": players.get(b, {}).get("pos", ""),
                      "inj": players.get(b, {}).get("inj"),
                      "proj": round(val(b), 2), "gap": round(val(b) - val(pid), 2), "set": b in set_now} for b in alts],
        })
    not_optimal = [players.get(p, {}).get("name", p) for p in set_now if p not in optimal_ids]

    settings = lg.get("settings") or {}
    report = {
        "league_id": lid, "name": lg.get("name"), "week": week,
        "as": {"rid": my_rid, "name": users.get(mine.get("owner_id"), "?"), "is_me": mine is me_roster},
        "teams": [{"rid": r["roster_id"], "name": users.get(r.get("owner_id"), f"Roster {r['roster_id']}")} for r in rosters],
        "field": field,
        "faab": {"budget": settings.get("waiver_budget"), "used": (mine.get("settings") or {}).get("waiver_budget_used", 0)},
        "field_size": len(rosters), "base": base, "holes": holes, "candidates": cands, "chop_name": chop_name,
        "slot_order": slot_order,
        "lineup": lineup_rows_out, "set_not_optimal": not_optimal, "lineup_set": bool(set_now),
    }
    # Sort by what the add is worth to this roster, not by raw projection —
    # otherwise streaming QBs crowd out a real upgrade in a 1-QB league.
    cands.sort(key=lambda c: (-c["delta"], -c["proj"]))
    report["rank_source"] = _season_cache["source"]
    report["candidates_all"] = cands            # for logging; UI shows WAIVER_TOP
    report["candidates"] = cands[:WAIVER_TOP]
    _waiver_cache[(lid, as_rid)] = (time.time(), report)
    if as_rid is None:
        log_fa_snapshot(report, len(rosters), users, rosters, budget)
        sync_bids(lid, week, users, rosters, budget)
    return report


# ----------------------------------------------------------------------------
# Bids database: the pre-bid board (fa_snapshots) and every claim, won or
# lost, joined to that board (bids). Field size is the season's time axis.
# ----------------------------------------------------------------------------

def connect_log():
    """The log DB in WAL mode with a generous busy timeout: the build loop,
    the waiver/bids sync and page reads all share it from different threads,
    and a slow backfill transaction was locking readers out."""
    con = sqlite3.connect(LOG_DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _db():
    con = connect_log()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS fa_snapshots (
            ts INTEGER, season TEXT, week INTEGER, league_id TEXT, field_size INTEGER,
            pid TEXT, name TEXT, pos TEXT, proj REAL, pos_rank INTEGER,
            starts_for INTEGER, of_teams INTEGER, tier INTEGER, max_delta REAL, max_swing REAL, chop_pool INTEGER,
            season_rank INTEGER);
        CREATE INDEX IF NOT EXISTS fa_snap_ix ON fa_snapshots (league_id, week, pid, ts);
        CREATE TABLE IF NOT EXISTS bids (
            tx_id TEXT PRIMARY KEY, season TEXT, week INTEGER, league_id TEXT, ts INTEGER,
            rid TEXT, owner TEXT, pid TEXT, name TEXT, pos TEXT, bid INTEGER, status TEXT,
            drop_pid TEXT, drop_name TEXT,
            faab_before INTEGER, share REAL, field_size INTEGER,
            proj REAL, pos_rank INTEGER, starts_for INTEGER, of_teams INTEGER, tier INTEGER,
            n_bidders INTEGER, second_bid INTEGER, winner INTEGER, season_rank INTEGER);
    """)
    for table in ("fa_snapshots", "bids"):        # migrate DBs created before season_rank
        if "season_rank" not in [r[1] for r in con.execute(f"PRAGMA table_info({table})")]:
            con.execute(f"ALTER TABLE {table} ADD COLUMN season_rank INTEGER")
    con.commit()
    return con


def log_fa_snapshot(report, field_size, users, rosters, budget):
    """One row per free agent on the board, at most hourly per league/week."""
    con = _db()
    lid, week = report["league_id"], report["week"]
    last = con.execute("SELECT MAX(ts) FROM fa_snapshots WHERE league_id=? AND week=?", (lid, week)).fetchone()[0]
    if last and time.time() - last < 3600:
        con.close()
        return
    ts = int(time.time())
    con.executemany("INSERT INTO fa_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (ts, CONFIG["season"], week, lid, field_size, c["id"], c["name"], c["pos"], c["proj"], c["pos_rank"],
         c["demand"]["starts"], c["demand"]["of"], c["tier"], c["max_delta"], c["max_swing"], int(c["chop_pool"]),
         c["season_rank"])
        for c in report["candidates_all"]])
    con.commit()
    con.close()


def week_of(ts):
    """NFL week a timestamp falls in (weeks roll on Sleeper's season start
    weekday, a Wednesday). A Wednesday-3am waiver run counts for the week
    that is about to be played, which is what the bids were for."""
    st = get("/state/nfl")
    start = time.mktime(time.strptime(st["season_start_date"], "%Y-%m-%d"))
    return max(1, int((ts - start) // (7 * 86400)) + 1)


def backfill_board(lid, run_claims, run_ts, rosters, users, players, con):
    """Write fa_snapshot rows for the players bid on in a past run, with
    demand computed against the rosters as they stood before that run."""
    wk = week_of(run_ts)                       # the week the pickups were FOR
    projections = load_projections(CONFIG["season"], wk, players)
    lg = get(f"/league/{lid}")
    scoring = lg.get("scoring_settings") or {}
    slots = lg.get("roster_positions") or []
    val = lambda p: proj_pts(p, projections, scoring) or 0.0
    adds = {c["pid"] for c in run_claims if c["status"] == "complete"}
    drops = {c["rid"]: c["dpid"] for c in run_claims if c["status"] == "complete" and c["dpid"]}
    then = []
    for r in rosters:
        ids = [p for p in (r.get("players") or []) if p not in adds]
        if r["roster_id"] in drops:
            ids.append(drops[r["roster_id"]])
        then.append({"roster_id": r["roster_id"], "players": ids})
    usable = set().union(*(ELIGIBLE.get(sl, {sl}) for sl in slots if sl not in NON_SLOTS))
    pos_rank = {}
    for pos in usable:
        ranked = sorted((pid for pid in projections if players.get(pid, {}).get("pos") == pos), key=lambda p: -val(p))
        pos_rank.update({pid: i + 1 for i, pid in enumerate(ranked)})
    base = {r["roster_id"]: sum(val(p) for p in best_lineup(r["players"], slots, val, players)) for r in then}
    srank = season_pos_rank(players)
    rows = []
    for pid in {c["pid"] for c in run_claims}:
        starts, best = 0, 0.0
        for r in then:
            lineup = best_lineup(r["players"] + [pid], slots, val, players)
            if pid in lineup:
                starts += 1
                best = max(best, sum(val(p) for p in lineup) - base[r["roster_id"]])
        of = len(then)
        p = players.get(pid, {})
        rows.append((run_ts - 1, CONFIG["season"], wk, lid, of, pid, p.get("name", pid), p.get("pos", ""),
                     round(val(pid), 2), pos_rank.get(pid), starts, of,
                     tier_of(starts / max(1, of), p.get("pos"), srank.get(pid)), round(best, 2), None, 0, srank.get(pid)))
    con.executemany("INSERT INTO fa_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()


def owner_names(lid, users, con):
    """{roster_id: owner} for every roster in the league, chopped ones too.
    `rosters` elsewhere is survivors only, which left eliminated bidders as
    "?". Sleeper keeps owner_id on an emptied roster; if that's ever missing,
    fall back to the name the calibration log recorded for the roster."""
    out = {}
    for r in get(f"/league/{lid}/rosters") or []:
        name = users.get(r.get("owner_id"))
        if name:
            out[r["roster_id"]] = name
    for rid, name in con.execute("SELECT rid, name FROM teams WHERE league_id=? ORDER BY week", (lid,)).fetchall():
        out.setdefault(int(rid), name)
    return out


def sync_bids(lid, week, users, rosters, budget):
    """Pull Sleeper's waiver claims for every week so far and upsert them
    with context from the latest snapshot taken before the claim ran."""
    if time.time() - _bids_last.get(lid, 0) < 900:
        return
    _bids_last[lid] = time.time()
    players = load_players()
    con = _db()
    rid_owner = owner_names(lid, users, con)
    claims = []
    for wk in range(1, week + 1):
        for t in get(f"/league/{lid}/transactions/{wk}") or []:
            if t.get("type") != "waiver" or not t.get("roster_ids"):
                continue
            adds = t.get("adds") or {}
            if not adds:
                continue
            pid = next(iter(adds))
            drops = t.get("drops") or {}
            dpid = next(iter(drops), None)
            claims.append({
                "tx_id": str(t["transaction_id"]), "week": wk, "ts": int(t["status_updated"] / 1000),
                "rid": t["roster_ids"][0], "pid": pid, "dpid": dpid,
                "bid": int((t.get("settings") or {}).get("waiver_bid") or 0), "status": t.get("status"),
            })
    if not claims:
        con.close()
        return
    # FAAB before each claim: budget minus this roster's wins in EARLIER runs
    # (claims in the same run were all placed against the same balance).
    won, run_won = {}, {}
    for run in sorted({c["ts"] // 600 for c in claims}):
        for c in [c for c in claims if c["ts"] // 600 == run]:
            c["faab_before"] = budget - won.get(c["rid"], 0)
            if c["status"] == "complete":
                run_won[c["rid"]] = run_won.get(c["rid"], 0) + c["bid"]
        for rid, v in run_won.items():
            won[rid] = won.get(rid, 0) + v
        run_won = {}
    # Runs with no pre-bid board (before this table existed): reconstruct it.
    # Rosters as they were = current rosters minus that run's completed adds,
    # plus its drops; then the demand analysis for the players bid on.
    for run in sorted({c["ts"] // 600 for c in claims}):
        run_claims = [c for c in claims if c["ts"] // 600 == run]
        run_ts = min(c["ts"] for c in run_claims)
        have = con.execute("SELECT COUNT(*) FROM fa_snapshots WHERE league_id=? AND ts<=? AND ts>?",
                           (lid, run_ts, run_ts - 7 * 86400)).fetchone()[0]
        if have:
            continue
        backfill_board(lid, run_claims, run_ts, rosters, users, players, con)

    # Clearing price and competition per (player, run).
    by_key = {}
    for c in claims:
        by_key.setdefault((c["pid"], c["ts"] // 600), []).append(c)
    rows = []
    for c in claims:
        group = sorted(by_key[(c["pid"], c["ts"] // 600)], key=lambda x: -x["bid"])
        second = group[1]["bid"] if len(group) > 1 else None
        snap = con.execute(
            "SELECT field_size, proj, pos_rank, starts_for, of_teams, tier, season_rank FROM fa_snapshots "
            "WHERE league_id=? AND pid=? AND ts<=? ORDER BY ts DESC LIMIT 1", (lid, c["pid"], c["ts"])).fetchone()
        if snap is None:  # no pre-bid board (e.g. week 1): fall back to the earliest one after
            snap = con.execute(
                "SELECT field_size, proj, pos_rank, starts_for, of_teams, tier, season_rank FROM fa_snapshots "
                "WHERE league_id=? AND pid=? ORDER BY ts LIMIT 1", (lid, c["pid"])).fetchone()
        fs, proj, prank, sf, of, tier, srk = snap if snap else (len(rosters), None, None, None, None, None, None)
        p = players.get(c["pid"], {})
        rows.append((c["tx_id"], CONFIG["season"], week_of(c["ts"]), lid, c["ts"], str(c["rid"]), rid_owner.get(c["rid"], f"Roster {c['rid']}"),
                     c["pid"], p.get("name", c["pid"]), p.get("pos", ""), c["bid"], c["status"],
                     c["dpid"], players.get(c["dpid"], {}).get("name") if c["dpid"] else None,
                     c["faab_before"], round(c["bid"] / c["faab_before"], 4) if c["faab_before"] else None, fs,
                     proj, prank, sf, of, tier, len(group), second, int(c["status"] == "complete"), srk))
    con.executemany("INSERT OR REPLACE INTO bids VALUES (" + ",".join("?" * 26) + ")", rows)
    con.commit()
    con.close()


def roster_names(lid):
    """{roster_id: display_name} straight from Sleeper, cached for the process."""
    if lid not in _roster_names:
        try:
            users = {u["user_id"]: u.get("display_name") or "?" for u in get(f"/league/{lid}/users")}
            _roster_names[lid] = {str(r["roster_id"]): users.get(r.get("owner_id"), f"Roster {r['roster_id']}")
                                  for r in get(f"/league/{lid}/rosters")}
        except Exception:
            return {}
    return _roster_names[lid]


@app.route("/api/history/<lid>")
def api_history(lid):
    """
    Each team's live points and projected final over the week, from the
    calibration log: one point per snapshot, downsampled to ~400 per team.
    """
    week = request.args.get("week", type=int)
    season = CONFIG["season"]
    if not LOG_DB.exists():
        return jsonify({"teams": [], "week": week})
    hit = _hist_cache.get((lid, week))
    if hit and time.time() - hit[0] < (90 if week is None or week == (_cache["data"] or {}).get("week") else 86400):
        return jsonify(hit[1])
    # One build at a time, and nobody queues behind it: waiting requests get
    # the stale copy (or an empty chart) instead of pinning a worker thread.
    if not _hist_lock.acquire(blocking=False):
        return jsonify(hit[1] if hit else {"week": week, "weeks": [], "league_id": lid, "teams": [], "building": True})
    try:
        payload = history_payload(lid, week, season)
        _hist_cache[(lid, week)] = (time.time(), payload)
    finally:
        _hist_lock.release()
    return jsonify(payload)


_hist_cache = {}
_hist_lock = threading.Lock()


def history_payload(lid, week, season):
    con = connect_log()
    try:
        con.executescript(TEAM_SNAPS_DDL)
    except sqlite3.OperationalError:          # nothing logged yet
        con.close()
        return {"week": week, "weeks": [], "league_id": lid, "teams": []}
    weeks = [w for (w,) in con.execute("SELECT DISTINCT week FROM leagues WHERE season=? AND league_id=? ORDER BY week",
                                       (season, lid)).fetchall()]
    if week is None:
        week = (weeks[-1] if weeks else None) or (_cache["data"] or {}).get("week") or 1
    rows = con.execute("SELECT ts, rid, pts, proj FROM team_snaps WHERE season=? AND league_id=? AND week=? ORDER BY ts",
                       (season, lid, week)).fetchall()
    if not rows:
        # Weeks logged before team_snaps existed: aggregate the players table
        # once and keep the result.
        raw = con.execute(
            "SELECT ts, rid, ROUND(SUM(act), 2), ROUND(SUM(act + proj * rem), 2) FROM players "
            "WHERE season=? AND week=? AND league_id=? GROUP BY ts, rid ORDER BY ts", (season, week, lid)).fetchall()
        con.executemany("INSERT OR REPLACE INTO team_snaps VALUES (?,?,?,?,?,?,?)",
                        [(season, week, lid, ts, rid, p, f) for ts, rid, p, f in raw])
        con.commit()
        rows = raw
    names = dict(con.execute("SELECT rid, name FROM teams WHERE season=? AND week=? AND league_id=?",
                             (season, week, lid)).fetchall())
    con.close()
    # Snapshots older than the teams table (Week 1) have no stored names, and
    # chopped teams have left the live field: ask Sleeper, whose rosters list
    # keeps eliminated teams. Cached per league; also written back to the log.
    missing = {rid for _, rid, *_ in rows} - set(names)
    if missing and not lid.startswith(("espn:", "manual:")):
        fetched = roster_names(lid)
        con = connect_log()
        con.executemany("INSERT OR REPLACE INTO teams VALUES (?,?,?,?,?)",
                        [(season, week, lid, rid, fetched[rid]) for rid in missing if rid in fetched])
        con.commit()
        con.close()
        names.update({rid: fetched[rid] for rid in missing if rid in fetched})
    live = next((l for l in (_cache["data"] or {}).get("leagues", []) if l["league_id"] == lid), None)
    for t in (live or {}).get("field") or []:
        names.setdefault(str(t["rid"]), t["name"])
    for t in (live or {}).get("standings") or []:
        names.setdefault(str(t["rid"]), t["name"])
    names.setdefault("me", "you")
    series = {}
    for ts, rid, pts, proj in rows:
        series.setdefault(rid, []).append([ts, pts, proj])
    # Trim the trailing flat run (post-game snapshots) so the axis ends when
    # the last live score moved, keeping one point past it. Projections keep
    # drifting after the games (Sleeper revises them), so key on points only.
    snaps = sorted({ts for ts, *_ in rows})
    by_ts = {}
    for ts, rid, pts, proj in rows:
        by_ts.setdefault(ts, {})[rid] = pts
    # Keep only "action" snapshots: the first, and any where some team's live
    # points moved since the previous kept one. Dead time between games
    # (where only projections twitch) drops out entirely.
    keep, prev = [], None
    for t in snaps:
        if prev is None or by_ts[t] != by_ts[prev]:
            keep.append(t)
            prev = t
    keep_set = set(keep)
    series = {rid: [pt for pt in v if pt[0] in keep_set] for rid, v in series.items()}
    step = max(1, max((len(v) for v in series.values()), default=1) // 400)
    my_rid = str((live or {}).get("my_rid", ""))
    opp = (live or {}).get("opp_name")
    teams = [{"rid": rid, "name": names.get(rid, f"Roster {rid}"), "me": rid == my_rid or rid == "me",
              "opp": names.get(rid) == opp,
              "series": v[::step] + ([v[-1]] if (len(v) - 1) % step else [])}
             for rid, v in series.items()]
    teams.sort(key=lambda t: -t["series"][-1][2])
    return {"week": week, "weeks": weeks, "league_id": lid, "mode": (live or {}).get("mode"), "teams": teams}


@app.route("/api/waivers/<lid>")
def api_waivers(lid):
    try:
        return jsonify(waiver_report(lid, request.args.get("as")))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/bids/<lid>")
def api_bids(lid):
    """Last run's results grouped by player, and per-owner behaviour."""
    if not LOG_DB.exists():
        return jsonify({"runs": [], "owners": []})
    con = connect_log()
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM bids WHERE league_id=? ORDER BY ts DESC, pid, bid DESC", (lid,)).fetchall()]
    con.close()
    runs = {}
    for r in rows:
        runs.setdefault(r["ts"] // 600, []).append(r)
    out_runs = []
    for k in sorted(runs, reverse=True)[:6]:
        by_p = {}
        for r in runs[k]:
            by_p.setdefault(r["pid"], {"name": r["name"], "pos": r["pos"], "tier": r["tier"], "proj": r["proj"], "season_rank": r["season_rank"],
                                       "starts_for": r["starts_for"], "of": r["of_teams"], "bids": []})
            by_p[r["pid"]]["bids"].append({"owner": r["owner"], "bid": r["bid"], "won": r["winner"],
                                          "share": r["share"], "drop": r["drop_name"]})
        players_ = sorted(by_p.values(), key=lambda x: -max(b["bid"] for b in x["bids"]))
        out_runs.append({"ts": min(r["ts"] for r in runs[k]), "week": runs[k][0]["week"],
                         "field_size": runs[k][0]["field_size"], "players": players_})
    owners = {}
    for r in rows:
        o = owners.setdefault(r["owner"], {"owner": r["owner"], "bids": 0, "won": 0, "spent": 0, "faab": None,
                                          "by_tier": {1: [], 2: [], 3: [], None: []}, "over": []})
        o["bids"] += 1
        o["won"] += r["winner"]
        o["spent"] += r["bid"] if r["winner"] else 0
        o["by_tier"].setdefault(r["tier"], []).append(r["share"] or 0)
        if r["winner"] and r["second_bid"] is not None and r["second_bid"] > 0:
            o["over"].append(r["bid"] / r["second_bid"])
    for o in owners.values():
        o["by_tier"] = {str(t or "?"): {"n": len(v), "avg_share": round(100 * sum(v) / len(v), 1)} for t, v in o["by_tier"].items() if v}
        o["avg_over"] = round(sum(o["over"]) / len(o["over"]), 2) if o["over"] else None
        o.pop("over")
    # Price tags: FAAB spent on each player over the season, every time he
    # clears waivers (dropped, chopped, re-bought). Also across all shared
    # leagues, for the running cross-league tally.
    tags = {}
    for r in sorted(rows, key=lambda r: r["ts"]):
        if r["winner"] and r["bid"] > 0:
            t = tags.setdefault(r["pid"], {"name": r["name"], "pos": r["pos"], "total": 0, "buys": []})
            t["total"] += r["bid"]
            t["buys"].append({"week": r["week"], "owner": r["owner"], "bid": r["bid"]})
    con = connect_log()
    everywhere = dict(con.execute(
        "SELECT pid, SUM(bid) FROM bids WHERE winner=1 AND season=? AND league_id IN (%s) GROUP BY pid"
        % ",".join("?" * len(CONFIG["shared_leagues"])), (CONFIG["season"], *CONFIG["shared_leagues"])).fetchall())
    con.close()
    for pid, t in tags.items():
        t["all_leagues"] = everywhere.get(pid, t["total"])
    leaders = sorted(tags.values(), key=lambda t: -t["total"])[:15]
    try:
        users = {u["user_id"]: u.get("display_name") for u in get(f"/league/{lid}/users")}
        out_ = {users.get(r.get("owner_id")) for r in get(f"/league/{lid}/rosters") if not r.get("players")}
    except Exception:
        out_ = set()
    for o in owners.values():
        o["chopped"] = o["owner"] in out_
    return jsonify({"runs": out_runs, "owners": sorted(owners.values(), key=lambda o: -o["spent"]), "leaders": leaders})


@app.route("/l/<lid>")
def league_page(lid):
    if lid not in CONFIG["shared_leagues"]:
        return Response("not shared", 404)
    return Response(PAGE.replace("<script>", f"<script>const LEAGUE_ID = {json.dumps(lid)};", 1),
                    mimetype="text/html")


@app.route("/healthz")
def healthz():
    return "ok"


@app.route("/tick")
def tick():
    """Unauthenticated poke that runs a build (and so a log snapshot). Hit by
    the game-window cron so the calibration log records without a viewer.
    ?waivers=1 also refreshes the waiver boards (pre-bid snapshots + bid sync)."""
    data, err = current_state()
    if err:
        return Response(f"error: {err}", 502)
    if request.args.get("waivers"):
        for lid in CONFIG["shared_leagues"]:
            try:
                waiver_report(lid)
            except Exception as exc:
                return Response(f"waivers error {lid}: {exc}", 502)
    return Response(f"ok wk{data['week']} {data['updated']}", mimetype="text/plain")


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

  /* chop-line rows: same scale as the h2h rows, coloured edge for danger */
  .chop{border-left:3px solid var(--rule);padding:7px 0 7px 12px;
        border-bottom:1px solid var(--rule);font-size:14px}
  .chop .l1,.chop .l2{display:flex;justify-content:space-between;gap:16px;align-items:baseline}
  .chop .l2{font-size:12.5px;margin-top:2px}
  .chop .lg b{font-weight:600;color:var(--ink)}
  .chop .big{font-size:16px;font-weight:600}
  .chop .r1{text-align:right;white-space:nowrap}
  .warnc{color:var(--warn);font-weight:500}
  .chop.danger{border-left-color:var(--short)}
  .chop.thin{border-left-color:var(--warn)}

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
  .detail tr.trow{cursor:pointer}
  .detail tr.trow:hover td{background:var(--bg)}
  .detail tr.trow .caret{display:inline-block;width:10px;font-size:10px;color:var(--mute)}
  .detail tr.tdetail > td{padding:2px 0 10px 22px;border-bottom:1px solid var(--rule)}
  .detail tr.tdetail h3{margin-top:4px}
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
  .chartsec{background:var(--panel);padding:10px 14px 12px;border-left:3px solid var(--rule)}
  .chartsec h3{font-size:12px;font-weight:600;color:var(--mute);margin:12px 0 6px}
  .chartsec table{font-size:13px}
  .chartsec td,.chartsec th{padding:3px 8px 3px 0}
  .chartsec tr.me td{font-weight:600}
  .chartsec tr.line td{border-bottom:2px solid var(--short)}
  tr.hole td{background:rgba(166,64,43,.07)}
  .chartwrap{margin:4px 0 14px}
  .chartbar{font-size:12.5px;margin-bottom:6px}
  .chartbar a{color:var(--mute);text-decoration:none}
  .chartbar select{font:inherit;font-size:12.5px;padding:2px 4px;border:1px solid var(--rule);background:var(--panel);color:var(--ink)}
  .chartbar a.on{color:var(--ink);font-weight:600;text-decoration:underline}
  .legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;margin-top:6px}
  .lg-item{cursor:pointer;white-space:nowrap}
  .lg-item input{vertical-align:middle;margin:0 2px 0 0}
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
let FAV = null;             // favourite league id, from the server
const open = new Set();   // league ids with the drill-down expanded; survives re-render
let showMinor = false;    // feed: show the full list instead of the last 12
const collapsed = new Set((() => { try { return JSON.parse(localStorage.getItem('collapsed') || '[]'); } catch(e){ return []; } })());
function toggleSection(id){
  collapsed.has(id) ? collapsed.delete(id) : collapsed.add(id);
  userToggled.add(id);
  try { localStorage.setItem('collapsed', JSON.stringify([...collapsed])); localStorage.setItem('toggled', JSON.stringify([...userToggled])); } catch(e){}
  tick();
}
const userToggled = new Set((() => { try { return JSON.parse(localStorage.getItem('toggled') || '[]'); } catch(e){ return []; } })());
function section(id, title, body, count, autoCollapse){
  // autoCollapse: fold by default (e.g. an empty feed) unless the user has toggled this section themselves.
  const c = userToggled.has(id) ? collapsed.has(id) : (autoCollapse ?? collapsed.has(id));
  return `<h2 class="sec" onclick="toggleSection('${id}')"><span class="caret">${c ? '▸' : '▾'}</span> ${title}${count !== undefined ? ` <span class="cnt">${count}</span>` : ''}</h2>${c ? '' : body}`;
}
function toggle(id){ open.has(id) ? open.delete(id) : open.add(id); tick(); }
const f2 = x => (x ?? 0).toFixed(2);
const inj = s => s ? ` <span class="${['Questionable'].includes(s) ? 'warnc' : 'short'}" style="font-size:11px">${({Questionable:'Q', Doubtful:'D', Out:'OUT', IR:'IR', PUP:'PUP', Sus:'SUS'})[s] || s}</span>` : '';
const tp = t => t ? `${t[0]} to play${t[1] ? ` (${t[1]} live)` : ''}` : '';
const pctCls = p => p >= 85 ? 'long' : (p < 60 ? 'short' : '');

function lineupTable(title, rows){
  if (!rows || !rows.length) return '';
  const left = r => r.rem >= 1 ? '<span class="pos">—</span>' : (r.rem <= 0 ? '<span class="pos">final</span>' : `<span style="color:var(--warn)">${Math.round(r.rem*60)}'${r.ot ? ' OT' : ''}</span>`);
  const tot = k => rows.reduce((a,r) => a + r[k], 0);
  return `<h3>${title} &middot; <span class="num">${f2(tot('act'))}</span> &rarr; <span class="num">${f2(tot('final'))}</span></h3>
    <table class="lineup"><tr><th>Pos</th><th>Player</th><th>Line</th><th class="r">Pts</th><th class="r">Proj final</th><th class="r">Left</th></tr>
    ${rows.map(r => `<tr>
      <td class="pos">${r.pos}</td><td>${r.name} <span class="pos">${r.team}</span>${inj(r.inj)}</td>
      <td class="pos">${r.line}</td>
      <td class="r num">${r.act ? f2(r.act) : '<span class="pos">0.00</span>'}</td>
      <td class="r num ${r.final > r.proj ? 'long' : (r.rem <= 0 && r.final < r.proj ? 'short' : '')}">${f2(r.final)}<span class="pos" style="font-size:11px"> /${r.proj.toFixed(1)}</span></td>
      <td class="r num">${left(r)}</td></tr>`).join('')}
    </table>`;
}

function benchBlock(l){
  const rows = l.bench_lineup || [];
  if (!rows.length) return '';
  const key = 'bench:' + l.league_id, isOpen = open.has(key);
  return `<h3 class="bench" onclick="event.stopPropagation();toggle('${key}')" style="cursor:pointer">Bench <span class="cnt">${rows.length}</span> · <span style="text-decoration:underline">${isOpen ? 'hide' : 'show'}</span></h3>
    ${isOpen ? lineupTable('Bench', rows).replace(/<h3>.*?<\/h3>/, '') : ''}`;
}

// ---- history chart ----
const hist = {};            // league_id -> {ts, data}
const chartSel = {};        // league_id -> Set of rids shown
const chartWeek = {};       // league_id -> week being viewed (undefined = current)
let chartMode = 'proj';     // 'proj' | 'pts'
const PALETTE = ['#1F6F4A','#A6402B','#2B5FA6','#B07A16','#6B3FA0','#0F8B8D','#C2185B','#5D6D1E',
                 '#8C5A2B','#3C3C8C','#A02B7A','#2B8C5A','#7A4A1F','#1F7A8C','#8C1F3C','#4A6B1F','#6B2B8C','#8C6B1F'];

function histFresh(lid){ const h = hist[lid]; return h && h.week === chartWeek[lid] && Date.now() - h.ts < 60000; }
async function loadHistory(lid){
  if (histFresh(lid)) return hist[lid].data;
  try {
    const wk = chartWeek[lid];
    const d = await (await fetch(`/api/history/${lid}${wk ? `?week=${wk}` : ''}`)).json();
    hist[lid] = {ts: Date.now(), week: wk, data: d};
    return d;
  } catch(e){ return null; }
}
function chartSetWeek(lid, wk){ chartWeek[lid] = wk; delete chartSel[lid]; tick(); }

function chartToggleTeam(lid, rid){
  const sel = chartSel[lid]; sel.has(rid) ? sel.delete(rid) : sel.add(rid); tick();
}

function historyChart(l, d){
  if (!d || !d.teams.length) return '<div class="pos">No history logged for this week yet.</div>';
  if (!chartSel[l.league_id]){
    // Default: pools -> you, the projected chop line and the two above it;
    // head-to-head -> you and this week's opponent; solo -> you.
    const bottom = d.mode === 'pool' ? [...d.teams].slice(-3).map(t => t.rid) : [];
    chartSel[l.league_id] = new Set([...bottom, ...d.teams.filter(t => t.me || t.opp).map(t => t.rid)]);
  }
  const sel = chartSel[l.league_id];
  const idx = chartMode === 'proj' ? 2 : 1;
  const shown = d.teams.filter(t => sel.has(t.rid));
  const W = 820, H = 300, L = 44, R = 46, T = 10, B = 28;
  const all = shown.flatMap(t => t.series);
  // Even spacing per snapshot ("game time"): dead time is already dropped
  // server-side, and real-time jumps > 30 min get a dotted marker.
  const times = [...new Set(d.teams.flatMap(t => t.series.map(p => p[0])))].sort((a, b) => a - b);
  const xi = new Map(times.map((t, i) => [t, i]));
  const x0 = 0, x1 = Math.max(1, times.length - 1);
  const ys = all.map(p => p[idx]);
  let y0 = Math.min(...ys, chartMode === 'pts' ? 0 : Infinity), y1 = Math.max(...ys);
  if (!isFinite(y0) || !isFinite(y1)) return '<div class="pos">Pick a team to plot.</div>';
  if (y1 - y0 < 10) y1 = y0 + 10;
  const pad = (y1 - y0) * 0.05; y0 -= pad; y1 += pad;
  const X = t => L + ((xi.get(t) ?? 0) - x0) / Math.max(1, x1 - x0) * (W - L - R);
  const Y = v => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const hh = t => { const dt = new Date(t * 1000); return dt.getHours() + ':' + String(dt.getMinutes()).padStart(2, '0'); };
  // axes: 5 y ticks; ~8 time labels; dotted markers where real time jumps
  let g = '';
  for (let i = 0; i <= 4; i++){ const v = y0 + (y1 - y0) * i / 4; g += `<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--rule)"/><text x="${L-6}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--mute)">${v.toFixed(0)}</text>`; }
  const every = Math.max(1, Math.round(times.length / 8));
  const dayOf = t => new Date(t * 1000).toLocaleDateString(undefined, {weekday: 'short'});
  times.forEach((t, i) => {
    if (i % every === 0 || i === times.length - 1) g += `<text x="${X(t)}" y="${H-8}" text-anchor="middle" font-size="11" fill="var(--mute)">${hh(t)}</text>`;
    if (i > 0 && t - times[i-1] > 1800) g += `<line x1="${X(t)}" x2="${X(t)}" y1="${T}" y2="${H-B}" stroke="var(--mute)" stroke-dasharray="2,4" opacity=".6"/><text x="${X(t)+3}" y="${T+10}" font-size="10" fill="var(--mute)">${dayOf(t)} ${hh(t)}</text>`;
  });
  const lines = shown.map(t => {
    const col = PALETTE[d.teams.indexOf(t) % PALETTE.length];
    let path = '', prev = null;
    for (const p of t.series){ path += (prev === null || p[0] - prev > 1800 ? 'M' : 'L') + `${X(p[0]).toFixed(1)},${Y(p[idx]).toFixed(1)} `; prev = p[0]; }
    const last = t.series[t.series.length - 1];
    return `<path d="${path}" fill="none" stroke="${col}" stroke-width="${t.me ? 2.5 : 1.5}" ${t.me ? '' : 'opacity=".85"'}/>
      <text x="${X(last[0]) + 4}" y="${Y(last[idx]) + 4}" font-size="11" fill="${col}">${last[idx].toFixed(1)}</text>`;
  }).join('');
  const legend = d.teams.map(t => { const col = PALETTE[d.teams.indexOf(t) % PALETTE.length], on = sel.has(t.rid);
    return `<label class="lg-item" style="opacity:${on ? 1 : .45}"><input type="checkbox" ${on ? 'checked' : ''} onclick="event.stopPropagation();chartToggleTeam('${l.league_id}','${t.rid}')"> <span style="color:${col}">■</span> ${t.name}${t.me ? ' (you)' : ''}</label>`; }).join('');
  return `<div class="chartwrap" onclick="event.stopPropagation()">
    <div class="chartbar"><span class="pos">week <select onchange="chartSetWeek('${l.league_id}',+this.value)">${(d.weeks || [d.week]).map(w => `<option value="${w}" ${w === d.week ? 'selected' : ''}>${w}</option>`).join('')}</select> · </span>
      <a href="#" class="${chartMode === 'proj' ? 'on' : ''}" onclick="chartMode='proj';tick();return false">projected final</a> ·
      <a href="#" class="${chartMode === 'pts' ? 'on' : ''}" onclick="chartMode='pts';tick();return false">live points</a>
      <span class="pos"> · <a href="#" onclick="chartSel['${l.league_id}']=new Set(${JSON.stringify(d.teams.map(t=>t.rid))});tick();return false">all</a> · <a href="#" onclick="chartSel['${l.league_id}']=new Set();tick();return false">none</a></span></div>
    <svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;font-family:'IBM Plex Mono',monospace">${g}${lines}</svg>
    <div class="legend">${legend}</div>
  </div>`;
}

let chartLeague = null;     // league_id shown in the chart section
function chartSection(pools){
  pools = pools.filter(l => l.mode !== 'error');
  if (!pools.length) return '';
  if (!chartLeague || !pools.some(l => l.league_id === chartLeague)) chartLeague = (pools.find(l => l.league_id === FAV) || pools[0]).league_id;
  const l = pools.find(x => x.league_id === chartLeague);
  if (collapsed.has('chart')) return section('chart', 'Chart', '');
  if (!histFresh(l.league_id)) loadHistory(l.league_id).then(() => tick());
  const picker = pools.length > 1
    ? `<select onchange="chartLeague=this.value;tick()">${pools.map(x => `<option value="${x.league_id}" ${x.league_id === chartLeague ? 'selected' : ''}>${x.name}</option>`).join('')}</select>`
    : `<b>${l.name}</b>`;
  const body = `<div class="chartsec">
    <div class="pick" style="margin-bottom:8px">League &nbsp;${picker}</div>
    ${histFresh(l.league_id) ? historyChart(l, hist[l.league_id].data) : '<div class="pos">loading…</div>'}
  </div>`;
  return section('chart', 'Chart', body);
}

// ---- waivers (main page only) ----
const waiv = {};            // `${league_id}|${as}` -> {ts, data}
let waivLeague = null;
const waivAs = {};          // league_id -> rid being analysed ('' = me)
const waivPos = {};         // league_id -> Set of positions shown (default: all but QB in 1-QB leagues)
const waivKey = lid => `${lid}|${waivAs[lid] || ''}`;
async function loadWaivers(lid){
  const w = waiv[waivKey(lid)];
  if (w && Date.now() - w.ts < 600000) return w.data;
  try { const d = await (await fetch(`/api/waivers/${lid}${waivAs[lid] ? `?as=${waivAs[lid]}` : ''}`)).json(); waiv[waivKey(lid)] = {ts: Date.now(), data: d}; return d; }
  catch(e){ return null; }
}
// ---- bids history ----
const bidsCache = {};
async function loadBids(lid){
  const b = bidsCache[lid];
  if (b && Date.now() - b.ts < 600000) return b.data;
  try { const d = await (await fetch(`/api/bids/${lid}`)).json(); bidsCache[lid] = {ts: Date.now(), data: d}; return d; }
  catch(e){ return null; }
}
const TIER = {1: 'endgame', 2: 'starter', 3: 'fill-in'};
function bidsBlock(lid){
  const b = bidsCache[lid];
  if (!(b && Date.now() - b.ts < 600000)) { loadBids(lid).then(() => tick()); return '<div class="pos">loading bids…</div>'; }
  const d = b.data;
  if (!d.runs.length) return '<div class="pos">No waiver claims recorded yet.</div>';
  const key = 'bids:' + lid, isOpen = open.has(key);
  let html = `<h3 class="bench" onclick="event.stopPropagation();toggle('${key}')" style="cursor:pointer">Bids · ${d.runs.length} run${d.runs.length > 1 ? 's' : ''} · <span style="text-decoration:underline">${isOpen ? 'hide' : 'show'}</span></h3>`;
  if (!isOpen) return html;
  const run = d.runs[0];
  const when = new Date(run.ts * 1000).toLocaleDateString(undefined, {weekday: 'short', month: 'short', day: 'numeric'});
  html += `<div class="pos" style="font-size:12.5px;margin:0 0 6px">Last run · ${when} · for week ${run.week} · ${run.field_size} teams. Winner bold; share = bid as % of the bidder's FAAB at the time.</div>
    <table><tr><th>Player</th><th>Tier</th><th class="r">Proj</th><th class="r">Starts for</th><th style="padding-left:14px">Bids (high → low)</th></tr>
    ${run.players.map(p => `<tr>
      <td>${p.name} <span class="pos">${p.pos}</span></td>
      <td class="pos">${TIER[p.tier] || '?'}${p.tier === 1 && p.season_rank ? ` <span style="font-size:11px">${p.pos}${p.season_rank}</span>` : ''}</td>
      <td class="r num pos">${p.proj == null ? '—' : p.proj.toFixed(1)}</td>
      <td class="r num pos">${p.starts_for == null ? '—' : `${p.starts_for}/${p.of}`}</td>
      <td class="pos" style="padding-left:14px;font-size:12.5px">${p.bids.map(x => `${x.won ? '<b class="long">' : ''}${x.owner} <span class="num">${x.bid}</span>${x.share != null ? `<span style="font-size:11px"> (${(x.share*100).toFixed(0)}%)</span>` : ''}${x.won ? '</b>' : ''}`).join(', ')}</td></tr>`).join('')}</table>`;
  if (d.leaders && d.leaders.length) html += `<h3>Price tags · FAAB spent on each player this season, every time he clears waivers</h3>
    <table><tr><th>Player</th><th class="r">Total</th><th style="padding-left:14px">Bought (week · owner · $)</th><th class="r" title="the same player across Paris, Degenerates and Red Queen's">All leagues</th></tr>
    ${d.leaders.map(t => `<tr><td>${t.name} <span class="pos">${t.pos}</span></td><td class="r num"><b>${t.total}</b></td>
      <td class="pos" style="padding-left:14px;font-size:12.5px">${t.buys.map(x => `wk${x.week} ${x.owner} <span class="num">${x.bid}</span>`).join(' → ')}</td>
      <td class="r num pos">${t.all_leagues > t.total ? t.all_leagues : '—'}</td></tr>`).join('')}</table>`;
  html += `<h3>Owner behaviour · all runs · avg share of FAAB bid, by tier</h3>
    <table><tr><th>Owner</th><th class="r">Bids</th><th class="r">Won</th><th class="r">Spent</th><th class="r">Endgame</th><th class="r">Starter</th><th class="r">Fill-in</th><th class="r" title="winning bid / second-highest bid">Overpay ×</th></tr>
    ${d.owners.map(o => `<tr>
      <td>${o.owner}${o.chopped ? ' <span class="short" style="font-size:11px">chopped</span>' : ''}</td><td class="r num">${o.bids}</td><td class="r num">${o.won}</td><td class="r num">${o.spent}</td>
      ${['1','2','3'].map(t => { const v = o.by_tier[t]; return `<td class="r num ${v && v.avg_share >= 15 ? 'short' : 'pos'}">${v ? `${v.avg_share}% <span style="font-size:11px">(${v.n})</span>` : '—'}</td>`; }).join('')}
      <td class="r num pos">${o.avg_over == null ? '—' : o.avg_over.toFixed(2)}</td></tr>`).join('')}</table>`;
  return html;
}

function waiverSection(pools){
  if (LEAGUE_ID || !pools.length) return '';
  if (!waivLeague || !pools.some(l => l.league_id === waivLeague)) waivLeague = (pools.find(l => l.league_id === FAV) || pools[0]).league_id;
  if (collapsed.has('waivers')) return section('waivers', 'Waivers', '');
  const w = waiv[waivKey(waivLeague)];
  if (!(w && Date.now() - w.ts < 600000)) loadWaivers(waivLeague).then(() => tick());
  const picker = `<select onchange="waivLeague=this.value;tick()">${pools.map(x => `<option value="${x.league_id}" ${x.league_id === waivLeague ? 'selected' : ''}>${x.name}</option>`).join('')}</select>`;
  const asPick = w && !w.data.error ? ` &nbsp; Analyse as &nbsp;<select onchange="waivAs['${waivLeague}']=this.value;tick()"><option value="">me</option>${w.data.teams.filter(t => t.rid !== (w.data.as.is_me ? w.data.as.rid : -1)).map(t => `<option value="${t.rid}" ${String(t.rid) === String(waivAs[waivLeague] || '') ? 'selected' : ''}>${t.name}</option>`).join('')}</select>` : '';
  let body = `<div class="pick" style="margin-bottom:8px">League &nbsp;${picker}${asPick}</div>`;
  if (!w) body += '<div class="pos">computing…</div>';
  else if (w.data.error) body += `<div class="err">${w.data.error}</div>`;
  else {
    const d = w.data, sgn = x => x > 0 ? 'long' : (x < 0 ? 'short' : 'pos');
    const who = d.as.is_me ? 'you' : `<b class="warnc">${d.as.name}</b>`;
    body += `<div class="pos num" style="margin-bottom:10px">week ${d.week} · everyone on optimal lineups · ${who} proj <b>${f2(d.base.proj)}</b>, chop <b class="${d.base.chop_pct >= 15 ? 'short' : ''}">${d.base.chop_pct}%</b>${d.faab.budget ? ` · FAAB left <b>${d.faab.budget - d.faab.used}</b> of ${d.faab.budget}` : ''}</div>`;
    const posCols = d.slot_order || [];
    body += `<h3>Field · everyone's optimal lineup by projection, by slot</h3>
      <table><tr><th class="rk">#</th><th>Team</th><th class="r">Proj</th><th class="r">Chop %</th><th class="r">FAAB</th>${posCols.map(c => `<th class="r">${c}</th>`).join('')}</tr>
      ${d.field.map((f, i) => `<tr class="${f.me ? 'me' : ''} ${i === d.field.length - 2 ? 'line' : ''}">
        <td class="rk num">${i+1}</td><td>${f.name}${f.me ? (d.as.is_me ? ' <span class="pos">(you)</span>' : ' <span class="warnc">(as)</span>') : ''}</td>
        <td class="r num">${f2(f.proj)}</td><td class="r num ${f.chop_pct >= 15 ? 'short' : ''}">${Math.round(f.chop_pct)}</td>
        <td class="r num ${d.faab.budget && f.faab < d.faab.budget * 0.25 ? 'short' : 'pos'}">${d.faab.budget ? f.faab : '—'}</td>
        ${posCols.map(c => { const v = f.split[c]; const col = d.field.map(x => x.split[c] || 0); const lo = [...col].sort((a,b)=>a-b)[Math.floor(col.length/3)]; return `<td class="r num ${v != null && v <= lo ? 'short' : 'pos'}">${v == null ? '—' : v.toFixed(1)}</td>`; }).join('')}</tr>`).join('')}</table>`;
    const gapCls = g => g >= -1.5 ? 'warnc' : 'pos';
    body += `<h3>${d.as.is_me ? 'Your' : d.as.name + "'s"} optimal lineup · ✓ = currently set on Sleeper · bench alternatives with projection gap</h3>
      ${!d.lineup_set ? '<div class="short" style="font-size:12.5px;margin-bottom:6px">No lineup set on Sleeper yet.</div>' : (d.set_not_optimal.length ? `<div class="warnc" style="font-size:12.5px;margin-bottom:6px">Currently starting but not in the optimal lineup: ${d.set_not_optimal.join(', ')}</div>` : '')}
      <table><tr><th>Slot</th><th>Starter</th><th class="r">Proj</th><th style="padding-left:18px">Bench options</th></tr>
      ${d.lineup.map(r => `<tr>
        <td class="pos">${r.slot.replace('SUPER_FLEX','SF')}</td>
        <td>${r.name} <span class="pos">${r.pos} ${r.team}</span>${inj(r.inj)}${r.set ? ' <span class="long">✓</span>' : (d.lineup_set ? ' <span class="short">not set</span>' : '')}</td>
        <td class="r num">${f2(r.proj)}</td>
        <td class="pos" style="padding-left:18px;font-size:12.5px">${r.alts.map(a => `${a.name}${inj(a.inj)} <span class="num">${a.proj.toFixed(1)}</span> <span class="num ${gapCls(a.gap)}">(${a.gap > 0 ? '+' : ''}${a.gap.toFixed(1)})</span>${a.set ? ' <span class="short">set</span>' : ''}`).join(' · ') || '—'}</td></tr>`).join('')}</table>`;
    body += `<h3>Holes · ${d.as.is_me ? 'your' : 'their'} starter in each slot vs the field's</h3>
      <table><tr><th>Slot</th><th>Starter</th><th class="r">Proj</th><th class="r">Rank</th><th class="r">Median</th><th class="r">Best</th><th class="r">vs median</th></tr>
      ${d.holes.map(h => `<tr class="${h.rank > d.field_size * 0.67 ? 'hole' : ''}"><td class="pos">${h.pos}</td><td>${h.who}</td><td class="r num">${f2(h.mine)}</td>
        <td class="r num ${h.rank > d.field_size * 0.67 ? 'short' : (h.rank <= 3 ? 'long' : '')}">${h.rank}/${h.of}</td>
        <td class="r num pos">${f2(h.median)}</td><td class="r num pos">${f2(h.best)}</td>
        <td class="r num ${sgn(h.gap)}">${h.gap > 0 ? '+' : ''}${f2(h.gap)}</td></tr>`).join('')}</table>`;
    body += bidsBlock(waivLeague);
    const allPos = [...new Set(d.candidates_all.map(c => c.pos))].sort((a,b) => ['QB','RB','WR','TE','K','DEF'].indexOf(a) - ['QB','RB','WR','TE','K','DEF'].indexOf(b));
    if (!waivPos[waivLeague]) waivPos[waivLeague] = new Set(allPos.filter(p => p !== 'QB' || d.slot_order.includes('SF') || d.slot_order.some(x => x.startsWith('QB2'))));
    const shownPos = waivPos[waivLeague];
    const posBar = allPos.map(p => `<label class="lg-item"><input type="checkbox" ${shownPos.has(p) ? 'checked' : ''} onclick="event.stopPropagation();(waivPos['${waivLeague}'].has('${p}') ? waivPos['${waivLeague}'].delete('${p}') : waivPos['${waivLeague}'].add('${p}'));tick()"> ${p}</label>`).join(' ');
    const cands = d.candidates_all.filter(c => shownPos.has(c.pos)).slice(0, 12);
    body += `<h3>Top free agents · best Δ to your lineup, whole wire screened · endgame by ${d.rank_source || 'ROS rank'}${d.chop_name ? ` · <span class="short">chop pool</span> = still on ${d.chop_name}'s roster until dropped` : ''} &nbsp; <span style="font-weight:400">${posBar}</span></h3>
      <table><tr><th>Player</th><th class="r">Proj</th><th class="r">You after</th><th class="r">Δ proj</th><th class="r">Chop after</th><th>Displaces</th><th style="padding-left:14px">Demand · who else starts him</th></tr>
      ${cands.map(c => `<tr class="${c.delta <= 0 ? 'done' : ''}">
        <td>${c.name} <span class="pos">${c.pos} ${c.team}</span>${c.tier === 1 ? ` <span class="long" style="font-size:11px">endgame ${c.pos}${c.season_rank}</span>` : (c.tier === 2 ? ` <span class="pos" style="font-size:11px">starter</span>` : '')}${c.chop_pool ? ` <span class="short" style="font-size:11px">chop pool</span>` : ''}</td>
        <td class="r num">${f2(c.proj)}</td><td class="r num">${f2(c.proj_after)}</td>
        <td class="r num ${sgn(c.delta)}">${c.delta > 0 ? '+' : ''}${f2(c.delta)}</td>
        <td class="r num ${c.chop_after < d.base.chop_pct ? 'long' : 'pos'}">${c.chop_after}%<span class="pos" style="font-size:11px"> (${(c.chop_after - d.base.chop_pct) > 0 ? '+' : ''}${(c.chop_after - d.base.chop_pct).toFixed(1)})</span></td>
        <td class="pos" style="font-size:12.5px">${c.starts ? (c.displaces.join(', ') || '—') : 'bench'}</td>
        <td class="pos" style="font-size:12.5px;padding-left:14px"><b class="${c.demand.starts >= c.demand.of / 2 ? 'short' : ''}">${c.demand.starts}/${c.demand.of}</b>${c.demand.top.length ? ' · ' + c.demand.top.map(w => `${w.name} <span class="num">${w.chop_swing.toFixed(1)}%</span>`).join(', ') : ''}</td></tr>`).join('')}</table>`;
  }
  return section('waivers', 'Waivers', `<div class="chartsec">${body}</div>`);
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
    ${proj.map((t,i) => {
      const key = `team:${l.league_id}:${t.rid}`, isOpen = open.has(key);
      return `<tr class="${t.rid === me ? 'me' : ''} ${i === n - 2 ? 'line' : ''} trow" onclick="event.stopPropagation();toggle('${key}')">
      <td class="rk num">${i+1}</td><td><span class="caret">${isOpen ? '▾' : '▸'}</span> ${t.name}</td>
      <td class="r num ${t.chop_pct >= 15 ? 'short' : ''}">${Math.round(t.chop_pct)}</td>
      <td class="r num">${f2(t.proj)}</td><td class="r num">${f2(t.pts)}</td>
      <td class="r num pos">${t.to_play[0]}${t.to_play[1] ? `<span style="color:var(--warn)"> ·${t.to_play[1]}</span>` : ''}</td>
      <td class="r num pos">${lrank[t.rid]}</td></tr>
      ${isOpen && t.lineup ? `<tr class="tdetail"><td colspan="7">${lineupTable(t.name, t.lineup)}</td></tr>` : ''}`;
    }).join('')}
    </table>
    ${lineupTable('Your lineup', l.lineup)}
    ${benchBlock(l)}
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
    ${benchBlock(l)}
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
  if (l.below) nb.push(`<span>↓ ${l.below.name} ${l.below.pts}</span>`);
  if (l.above) nb.push(`<span>↑ ${l.above.name} ${l.above.pts}</span>`);
  const mc = m <= 0 ? 'short' : (m < 8 ? 'warnc' : 'long');
  return `<div class="chop ${cls}" onclick="toggle('${l.league_id}')">
    <div class="l1">
      <span class="lg">${l.name}${l.best_ball ? ' <span class="pos">· best ball</span>' : ''} <span class="pos">· proj <b>${l.proj_rank} of ${size}</b> · live ${l.rank}</span></span>
      <span class="num r1"><span class="big ${mc}">${m > 0 ? '+' : ''}${m.toFixed(2)}</span> <span class="pos">${verdict} · you ${l.my_points.toFixed(2)}</span></span>
    </div>
    <div class="l2 num pos">
      <span><span class="${pctCls(l.survive_pct)}">survive ${Math.round(l.survive_pct)}%</span> · ${l.sens.toFixed(2)}%/pt · ${l.my_to_play[0]} v ${pr.to_play[0]} to play</span>
      <span>${nb.join(' ')} · proj final ${l.my_proj.toFixed(2)} · ${projLine}</span>
    </div>
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
    const dfmt = (p, tag) => {
      const v = (p.deltas || {})[tag]; if (v == null) return '';
      const m = (p.mults || {})[tag];
      const total = `<span class="num">${v > 0 ? '+' : ''}${v.toFixed(1)}</span>`;
      return m ? ` <span class="pos num" style="font-size:11.5px">${m.base > 0 ? '+' : ''}${m.base.toFixed(1)}×${m.mult}</span> = ${total}` : ` ${total}`;
    };
    const tagList = side => { const seen = new Set(); return e.players.flatMap(p => p[side].map(t => `${t}${dfmt(p, t)}`)).filter(x => !seen.has(x) && seen.add(x)).join('<br>'); };
    const fors = tagList('for'), agst = tagList('against');
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
  d.teams.forEach(t => { t.field = d.field; });
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
        .map(p => ({...p, for: mine.has(p.id) ? ['you'] : [], against: theirs.has(p.id) ? [me.proj_ref.name] : [],
                    deltas: {you: (p.deltas || {})._, [me.proj_ref.name]: (p.deltas || {})._}}))}))
      .filter(e => e.players.length);
    html += section('feed', 'Feed', feedRows(feed), feed.length, !feed.length);
    html += chartSection([me]);
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
  FAV = d.favorite;

  const pools = d.leagues.filter(l => l.mode === 'pool').sort((a,b) => a.survive_pct - b.survive_pct);
  const h2h   = d.leagues.filter(l => l.mode === 'h2h');
  const manual= d.leagues.filter(l => l.mode === 'manual');
  const errors= d.leagues.filter(l => l.mode === 'error');

  let html = errors.map(l => `<div class="err">${l.name}: ${l.error}</div>`).join('');
  if (pools.length) html += section('pools', 'Guillotine', pools.map(chopCard).join(''));
  if (h2h.length) html += section('h2h', 'Head to head', h2h.map(h2hRow).join(''));
  if (manual.length) html += section('solo', 'Solo', manual.map(manualRow).join(''));
  html += section('feed', 'Feed', feedRows(d.feed), (d.feed || []).length, !(d.feed || []).length);
  const bookHtml = `<table>
    <tr><th>Player</th><th class="r">Pts</th><th class="r">Proj final</th><th class="r" title="root x remaining projection: swing still on the table">Impact</th><th class="r" title="pp of survival/win per fantasy point, summed over leagues">Root /pt</th><th>Leagues</th></tr>
    ${bookRows(d.book)}</table>`;
  const gamesLive = d.book.some(p => p.rem > 0 && p.rem < 1);
  html += section('book', 'Exposure', bookHtml, d.book.length, !gamesLive);
  html += waiverSection(pools);
  html += chartSection(d.leagues);
  if (manual.length){
    const bad = manual.flatMap(m => m.unmatched);
    if (bad.length) html += `<div class="err">Unmatched names in manual.json: ${bad.join(', ')}. Fix the spelling to fold them into exposure.</div>`;
    manual.filter(m => m.stale_week).forEach(m => { html = `<div class="err">${m.name}: manual.json is still the week ${m.stale_week} lineup — it's week ${d.week}. Update starters + multipliers and redeploy.</div>` + html; });
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
