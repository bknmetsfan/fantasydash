# fantasydash

A live dashboard for Chris's fantasy football leagues: three Sleeper guillotine
leagues (Paris in 1795v2, Degenerates, The Red Queen's League), two Sleeper dynasty
h2h leagues, one ESPN league and an eliminator.gg team tracked by hand.

Live at https://fantasydash.fly.dev (basic auth, any username, password = the
`BLOTTER_PASSWORD` Fly secret). Leaguemate views need no login:
`/l/1400335104223485952` (Paris), `/l/1389721132256473088` (Degenerates),
`/l/1389387283332874240` (Red Queen's).

## What's on the page
- **Guillotine cards**: live points, projected final, chop line and survival odds
  (Monte Carlo over remaining players). The drill-down shows the field with chop %,
  FAAB left, lineups, and in Degenerates each team's projected final with its best
  1–3 "get out of jail" bench swaps. Paris also tracks the weekly high score.
- **Head to head**: win %, plus the vs-median game and Max PF in the dynasty leagues.
- **Feed**: ESPN play-by-play for your players, priced in each league's scoring;
  benched players optional.
- **Exposure**: per player, how much each fantasy point matters across leagues.
- **Waivers** (main page only): optimal lineup, slot holes, field FAAB, free agents
  ranked by Δ to your lineup this week and over three weeks (bye-aware), rest-of-season
  ranks, demand from other teams, bids history and price tags.
- **Chart**: projected final, live points or chop % per team over the week, with a
  hover tooltip.

## Files
- `blotter.py` — everything: data fetching, simulation, Flask app and the page
  (inline HTML/JS in `PAGE`). Settings in the `CONFIG` dict near the top.
- `manual.json` — the eliminator.gg lineup; update weekly (`"week"`, eight starters,
  `"mult": 1.5` for FLEX and "unique" players). A banner shows when it's stale.
- `rankings/` — rest-of-season projection exports (FanDuel Research), one source per
  filename prefix; gitignored, see `rankings/README.md`. Blended with Sleeper.
- `title_odds.py` — rough title odds by simulating the rest of the season (chops +
  FAAB auctions). Script only: `.venv/bin/python title_odds.py [league_id ...]`.
- `espn.json` — ESPN cookies (gitignored, local only; on Fly they're secrets).
- `handoff/` — session notes; the newest file is the current state.

## Run locally
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PORT=5050 .venv/bin/python blotter.py      # macOS AirPlay holds :5000
```
Without `espn.json` the ESPN league is skipped. The calibration log is
`blotter.sqlite` (local) or `/data/blotter.sqlite` on Fly.

## Deploy
```
git push
flyctl deploy --app fantasydash --ha=false
```
`flyctl deploy` ships the local tree, so files in `rankings/` (not in git) go too —
deploy from the machine that has them. Fly app `fantasydash`, region ewr, 3 GB volume
`blotter_data` at `/data`, auto-stop on. Secrets: `BLOTTER_PASSWORD`, `ESPN_S2`,
`SWID`, `ESPN_LEAGUES`. If flyctl says `no access token available`, run
`flyctl auth login`.

`.github/workflows/tick.yml` hits `/tick` during game windows (keeps the log going
without a viewer) and Tue/Wed nights with `?waivers=1` (pre-bid boards + bid sync).

## Weekly routine
1. After Monday night: `curl -u x:$BLOTTER_PASSWORD 'https://fantasydash.fly.dev/tick?waivers=1'`,
   then check the Waivers boards. Manual-chop leagues (Degenerates, Red Queen's) need
   `CONFIG["pending_chop"]` only if a chop is decided but not yet executed in Sleeper.
2. Optional: drop a fresh FanDuel ROS export in `rankings/` (delete the old one) and deploy.
3. Before Sunday: update `manual.json` with the eliminator lineup and deploy.
