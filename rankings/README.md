Rest-of-season projection exports, one **source per filename prefix**:

    fanduel_ros_2026-09-29_qb.csv     -> source "fanduel", labelled "FanDuel 09-29"
    fanduel_ros_2026-09-29_flex.csv   -> same source (files with one prefix are merged)
    fantasypros_ros_2026-10-06.csv    -> source "fantasypros"

Each source is ranked by position (explicit ranks like `WR12` win, otherwise
points: `fantasy` / `FPTS` / `PTS` column). Sleeper's remaining weekly
projections are always a source too. The endgame tier (QB12 / RB10 / WR15 /
TE5) uses the blend: mean positional rank across the sources that list the
player, re-ranked. When refreshing a source, delete its old files (same
prefix) so stale rows don't mix in, and put the export date in the name.
