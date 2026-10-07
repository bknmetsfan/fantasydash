"""
Rough title odds for a guillotine league: simulate the rest of the season.

Each simulated week:
  1. every surviving team plays its optimal lineup by that week's Sleeper
     projection (byes come through as 0), each starter's score drawn from a
     gamma around the projection (blotter.CV per position);
  2. the lowest score is chopped; that roster joins the free-agent pool;
  3. a FAAB auction for the best pool players: every team bids on the gain
     to its lineup over the next three weeks, spending a share of its
     remaining budget that grows with the gain and as the field shrinks.

Title = last team standing. Simplifying assumptions (first cut):
  - one chop per week until one team remains;
  - no injuries beyond what the projections already carry, no trades;
  - rosters only grow (nobody drops; the bench doesn't affect lineups);
  - bid share = (gain / 20)^1.5 of remaining FAAB, capped, times urgency
    (1 + 2 x share of the field already gone) and owner noise (lognormal).
    Fitted by eye to Weeks 1-4: +9/wk endgame adds went for ~30-50% of
    budget, +5 starters 10-20%, +2 fill-ins a few %.

    .venv/bin/python title_odds.py [league_id ...]
"""
import sys
import time

import numpy as np

import blotter as b

AUCTION_PER_WEEK = 8       # pool players auctioned after each chop
HORIZON = 3                # weeks of projection a bidder values
POOL_START = 60            # current free agents kept in the pool


def league_inputs(lid):
    players = b.load_players()
    season = b.CONFIG["season"]
    week = b.get("/state/nfl").get("week") or 1
    lg = b.get(f"/league/{lid}")
    sc, slots = lg["scoring_settings"], lg["roster_positions"]
    users = {u["user_id"]: u.get("display_name") or "?" for u in b.get(f"/league/{lid}/users")}
    budget = (lg.get("settings") or {}).get("waiver_budget") or 0
    rosters = [r for r in b.get(f"/league/{lid}/rosters") if r.get("players")]
    weeks = list(range(week, b.LAST_WEEK + 1))
    qs = "&".join(f"position[]={p}" for p in ("QB", "RB", "WR", "TE", "K", "DEF"))
    proj = {week: b.load_projections(season, week, players)}      # this week: injury-aware
    for w in weeks[1:]:
        proj[w] = {r["player_id"]: r.get("stats") or {}
                   for r in b.get(f"/{season}/{w}?season_type=regular&{qs}", base=b.PROJ_BASE)}
    usable = set().union(*(b.ELIGIBLE.get(s, {s}) for s in slots if s not in b.NON_SLOTS)) & b.WAIVER_POS
    rostered = {p for r in rosters for p in r["players"]}
    ids = set(rostered) | {p for w in weeks for p in proj[w]
                           if p not in rostered and players.get(p, {}).get("pos") in usable}
    pts = {p: np.array([b.proj_pts(p, proj[w], sc) or 0.0 for w in weeks]) for p in ids}
    pool = sorted((p for p in ids if p not in rostered), key=lambda p: -pts[p][:HORIZON].mean())[:POOL_START]
    return {"name": lg.get("name"), "week": week, "weeks": weeks, "slots": slots, "players": players,
            "pts": pts, "pool": pool, "budget": budget,
            "teams": [{"rid": r["roster_id"], "name": users.get(r.get("owner_id"), f"Roster {r['roster_id']}"),
                       "players": [p for p in r["players"] if p in pts],
                       "faab": budget - ((r.get("settings") or {}).get("waiver_budget_used") or 0)} for r in rosters]}


def simulate_season(L, sims=400, seed=1, auction=True, pending_waivers=False, shut_out=()):
    """auction=False: frozen rosters, the no-waivers baseline.
    pending_waivers: run this week's (not yet processed) waiver auction
    before the first week is played; teams in shut_out win nothing in it."""
    rng = np.random.default_rng(seed)
    players, slots, pts, weeks = L["players"], L["slots"], L["pts"], L["weeks"]
    cv = {p: b.CV.get(players.get(p, {}).get("pos"), 0.7) for p in pts}
    rids = [t["rid"] for t in L["teams"]]
    n0 = len(rids)
    lineup = lambda roster, val: b.best_lineup(list(roster), slots, val.__getitem__, players)
    title = dict.fromkeys(rids, 0)
    exit_week = {r: [] for r in rids}
    alive_after = {r: np.zeros(len(weeks) + 1) for r in rids}
    spent = dict.fromkeys(rids, 0.0)
    def run_auction(ros, faab, pool, alive, hz, exclude=()):
        hv = {p: float(pts[p][hz].mean()) for p in pool | set().union(*(ros[r] for r in alive))}
        total = lambda r, extra=None: sum(hv[p] for p in lineup(ros[r] | ({extra} if extra else set()), hv))
        base = {r: total(r) for r in alive}
        urgency = 1 + 2 * (1 - len(alive) / n0)
        for p in sorted(pool, key=lambda p: -hv[p])[:AUCTION_PER_WEEK]:
            bids = []
            for r in alive:
                if r in exclude:
                    continue
                gain = total(r, p) - base[r]
                if gain < 0.5:
                    continue
                share = min(0.8, (gain / 20) ** 1.5) * urgency * rng.lognormal(0, 0.3)
                bids.append((min(faab[r], round(faab[r] * min(share, 1.0))), rng.random(), r))
            if not bids:
                continue
            bid, _, win = max(bids)
            faab[win] -= bid
            spent[win] += bid
            ros[win].add(p)
            pool.discard(p)
            base[win] = total(win)

    for _ in range(sims):
        ros = {t["rid"]: set(t["players"]) for t in L["teams"]}
        faab = {t["rid"]: float(t["faab"]) for t in L["teams"]}
        pool = set(L["pool"])
        alive = list(rids)
        if pending_waivers and auction:
            run_auction(ros, faab, pool, alive, slice(0, HORIZON), exclude=set(shut_out))
        for wi, w in enumerate(weeks):
            if len(alive) == 1:
                break
            val = {p: pts[p][wi] for s in (ros[r] for r in alive) for p in s}
            scores = {}
            for r in alive:
                lu = lineup(ros[r], val)
                m = np.array([val[p] for p in lu])
                c = np.array([cv[p] for p in lu])
                k = 1 / c ** 2
                scores[r] = float(np.sum(np.where(m > 0, rng.gamma(k, np.maximum(m, 1e-9) / k), 0.0)))
            out = min(alive, key=scores.get)
            alive.remove(out)
            exit_week[out].append(w)
            pool |= ros.pop(out)
            for r in alive:
                alive_after[r][wi] += 1
            if len(alive) == 1:
                break
            # Waivers for next week: bid on the gain over the next HORIZON weeks.
            if auction:
                run_auction(ros, faab, pool, alive, slice(wi + 1, min(wi + 1 + HORIZON, len(weeks))))
        title[alive[0]] += 1
    out = []
    for t in L["teams"]:
        r = t["rid"]
        now = {p: pts[p][0] for p in t["players"]}
        nxt = {p: float(pts[p][:HORIZON].mean()) for p in t["players"]}
        out.append({"rid": r, "name": t["name"], "faab": t["faab"],
                    "proj_now": round(sum(now[p] for p in lineup(t["players"], now)), 1),
                    "proj_3wk": round(sum(nxt[p] for p in lineup(t["players"], nxt)), 1),
                    "title_pct": round(100 * title[r] / sims, 1),
                    "chop_now_pct": round(100 * sum(1 for w in exit_week[r] if w == weeks[0]) / sims, 1),
                    "alive_4wk_pct": round(100 * alive_after[r][min(3, len(weeks) - 1)] / sims, 1),
                    "avg_exit_week": round(float(np.mean(exit_week[r])), 1) if exit_week[r] else None,
                    "avg_faab_spent": round(spent[r] / sims)})
    return sorted(out, key=lambda x: -x["title_pct"])


if __name__ == "__main__":
    lids = sys.argv[1:] or b.CONFIG["shared_leagues"]
    for lid in lids:
        t0 = time.time()
        L = league_inputs(lid)
        res = simulate_season(L, sims=int(__import__("os").environ.get("SIMS", 400)))
        print(f"\n{L['name']} — {len(L['teams'])} alive, from week {L['week']}  ({time.time() - t0:.0f}s)")
        print(f"  {'team':18} {'title%':>7} {'alive4w%':>8} {'chop now%':>9} {'proj now':>8} {'proj 3wk':>8} {'FAAB':>5} {'exp spend':>9}")
        for x in res:
            me = "*" if x["name"] == "ChrisPL" else ""
            print(f"  {x['name'][:17] + me:18} {x['title_pct']:7.1f} {x['alive_4wk_pct']:8.1f} {x['chop_now_pct']:9.1f} "
                  f"{x['proj_now']:8.1f} {x['proj_3wk']:8.1f} {x['faab']:5} {x['avg_faab_spent']:9}")
