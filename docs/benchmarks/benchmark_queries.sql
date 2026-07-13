-- ============================================================================
-- Database Performance Benchmark Suite
-- 20 Query Patterns for Measuring PostgreSQL Performance at 100M+ Rows
-- ============================================================================
-- Usage:
--   psql -d football_prediction -f benchmark_queries.sql -o benchmark_results.txt
--   # Or time individual queries:
--   \timing on
--   -- (run query)
-- ============================================================================

-- ══════════════════════════════════════════════════════════════════════════════
-- CATEGORY 1: Core Match Queries (5 patterns)
-- ══════════════════════════════════════════════════════════════════════════════

-- Q1: Upcoming matches (dashboard — uses partial index ix_matches_upcoming)
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.id, m.match_date, ht.name AS home_team, at.name AS away_team,
       s.name AS stadium, r.full_name AS referee
FROM matches m
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams at ON at.id = m.away_team_id
LEFT JOIN stadiums s ON s.id = m.stadium_id
LEFT JOIN referees r ON r.id = m.referee_id
WHERE m.status = 'scheduled'
ORDER BY m.match_date ASC
LIMIT 20;

-- Q2: Team's recent matches (uses covering indexes)
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.id, m.match_date, m.home_team_id, m.away_team_id,
       m.home_goals, m.away_goals, m.result, m.competition_id
FROM matches m
WHERE m.home_team_id = 42 OR m.away_team_id = 42
ORDER BY m.match_date DESC
LIMIT 10;

-- Q3: Match detail with all joins
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.*, ms.*, w.*, lo.*, la.*
FROM matches m
LEFT JOIN match_statistics ms ON ms.match_id = m.id
LEFT JOIN weather w ON w.match_id = m.id
LEFT JOIN odds lo ON lo.match_id = m.id AND lo.source = 'Pinnacle'
LEFT JOIN lineups la ON la.match_id = m.id
WHERE m.id = 50000000
ORDER BY lo.timestamp DESC
LIMIT 1;

-- Q4: League-season match list (uses composite index)
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.match_date, ht.name AS home, at.name AS away,
       m.home_goals, m.away_goals, m.result
FROM matches m
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams at ON at.id = m.away_team_id
WHERE m.competition_id = 10 AND m.season_id = 123
  AND m.status = 'finished'
ORDER BY m.match_date ASC;

-- Q5: Date range scan (uses BRIN index)
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT COUNT(*), AVG(m.home_goals), AVG(m.away_goals)
FROM matches m
WHERE m.match_date BETWEEN '2024-01-01' AND '2024-12-31'
  AND m.result IS NOT NULL;

-- ══════════════════════════════════════════════════════════════════════════════
-- CATEGORY 2: Team Analytics (5 patterns)
-- ══════════════════════════════════════════════════════════════════════════════

-- Q6: Team Elo history (uses partial index by side)
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT e.match_id, e.elo_before, e.elo_after, e.elo_change
FROM team_elo_history e
WHERE e.team_id = 42 AND e.side = 'home'
ORDER BY e.match_id DESC
LIMIT 50;

-- Q7: Team form over season
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT f.match_id, f.last_5_ppg, f.last_5_goals_scored,
       f.last_5_goals_conceded, f.season_ppg
FROM team_form f
WHERE f.team_id = 42
ORDER BY f.match_id DESC
LIMIT 38;

-- Q8: Team xG trend
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT x.match_id, x.xg, x.xg_open_play, x.xg_set_piece, x.xa
FROM team_xg_history x
WHERE x.team_id = 42 AND x.source = 'opta'
ORDER BY x.match_id DESC
LIMIT 38;

-- Q9: Match outcome prediction features (all team stats for a match)
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.id, m.match_date,
       he.elo_before AS home_elo, ae.elo_before AS away_elo,
       hf.last_5_ppg AS home_form, af.last_5_ppg AS away_form,
       hx.xg AS home_xg, ax.xg AS away_xg
FROM matches m
JOIN team_elo_history he ON he.team_id = m.home_team_id AND he.match_id = m.id
JOIN team_elo_history ae ON ae.team_id = m.away_team_id AND ae.match_id = m.id
JOIN team_form hf ON hf.team_id = m.home_team_id AND hf.match_id = m.id
JOIN team_form af ON af.team_id = m.away_team_id AND af.match_id = m.id
JOIN team_xg_history hx ON hx.team_id = m.home_team_id AND hx.match_id = m.id
JOIN team_xg_history ax ON ax.team_id = m.away_team_id AND ax.match_id = m.id
WHERE m.competition_id = 10 AND m.season_id = 123
ORDER BY m.match_date DESC
LIMIT 100;

-- Q10: Team vs team head-to-head
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.match_date, m.home_goals, m.away_goals, m.result
FROM matches m
WHERE ((m.home_team_id = 42 AND m.away_team_id = 7)
   OR (m.home_team_id = 7 AND m.away_team_id = 42))
  AND m.result IS NOT NULL
ORDER BY m.match_date DESC
LIMIT 10;

-- ══════════════════════════════════════════════════════════════════════════════
-- CATEGORY 3: Betting Analysis (5 patterns)
-- ══════════════════════════════════════════════════════════════════════════════

-- Q11: Value bets for upcoming matches
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.id, m.match_date,
       p.prob_home, o.odds_home,
       ROUND((p.prob_home * o.odds_home - 1)::numeric, 4) AS ev
FROM matches m
JOIN predictions p ON p.match_id = m.id AND p.model_name = 'ensemble'
JOIN odds o ON o.match_id = m.id AND o.source = 'Pinnacle'
WHERE m.status = 'scheduled'
  AND (p.prob_home * o.odds_home - 1) > 0.05
ORDER BY ev DESC
LIMIT 20;

-- Q12: Model performance across all matches
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT p.model_name,
       COUNT(*) AS total,
       AVG(CASE WHEN p.predicted_result = m.result THEN 1.0 ELSE 0.0 END) AS accuracy,
       AVG(p.confidence) AS avg_conf
FROM predictions p
JOIN matches m ON m.id = p.match_id AND m.result IS NOT NULL
WHERE p.prob_home IS NOT NULL
GROUP BY p.model_name
ORDER BY accuracy DESC;

-- Q13: Betting strategy P&L
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT br.strategy,
       COUNT(*) AS total_bets,
       SUM(CASE WHEN br.won THEN 1 ELSE 0 END) AS wins,
       ROUND(AVG(br.roi_pct)::numeric, 2) AS avg_roi,
       ROUND(SUM(br.profit)::numeric, 2) AS total_profit
FROM betting_results br
WHERE br.won IS NOT NULL
GROUP BY br.strategy
ORDER BY total_profit DESC;

-- Q14: Closing line value analysis
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT clv.bookmaker, clv.outcome,
       COUNT(*) AS n,
       ROUND(AVG(clv.clv)::numeric, 4) AS avg_clv,
       ROUND(AVG(clv.opening_price)::numeric, 2) AS avg_open,
       ROUND(AVG(clv.closing_price)::numeric, 2) AS avg_close
FROM closing_line_values clv
GROUP BY clv.bookmaker, clv.outcome
ORDER BY avg_clv DESC;

-- Q15: Best value bets by expected value
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT ev.match_id, ev.bookmaker,
       ROUND(ev.ev_home::numeric, 4) AS ev_home,
       ROUND(ev.ev_draw::numeric, 4) AS ev_draw,
       ROUND(ev.ev_away::numeric, 4) AS ev_away,
       ev.recommended_bet
FROM expected_value_bets ev
WHERE ev.recommended_ev > 0.10
ORDER BY ev.recommended_ev DESC
LIMIT 50;

-- ══════════════════════════════════════════════════════════════════════════════
-- CATEGORY 4: Aggregation & Reporting (5 patterns)
-- ══════════════════════════════════════════════════════════════════════════════

-- Q16: League standings (should use materialized view)
-- Compare: direct query vs mv_league_standings
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT team_id,
       SUM(played) AS played,
       SUM(wins) AS wins,
       SUM(draws) AS draws,
       SUM(losses) AS losses,
       SUM(goals_for) AS goals_for,
       SUM(goals_against) AS goals_against,
       SUM(points) AS points
FROM mv_league_standings
WHERE competition_id = 10 AND season_id = 123
GROUP BY team_id
ORDER BY points DESC;

-- Q17: Attendance trends by competition-season
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT m.competition_id, m.season_id,
       COUNT(*) AS matches,
       ROUND(AVG(m.attendance)::numeric, 0) AS avg_attendance,
       SUM(m.home_goals + m.away_goals) AS total_goals
FROM matches m
WHERE m.attendance IS NOT NULL
  AND m.result IS NOT NULL
GROUP BY m.competition_id, m.season_id
ORDER BY m.competition_id, m.season_id;

-- Q18: Referee home-bias analysis
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT r.full_name AS referee,
       COUNT(*) AS matches_officiated,
       ROUND(AVG(CASE WHEN m.result = 'H' THEN 1.0 ELSE 0.0 END)::numeric, 3) AS home_win_rate,
       ROUND(AVG(CASE WHEN m.result = 'D' THEN 1.0 ELSE 0.0 END)::numeric, 3) AS draw_rate
FROM matches m
JOIN referees r ON r.id = m.referee_id
WHERE m.result IS NOT NULL
  AND m.referee_id IS NOT NULL
GROUP BY r.id, r.full_name
HAVING COUNT(*) > 10
ORDER BY home_win_rate DESC;

-- Q19: Player performance over time
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT pms.match_id, pms.minutes_played, pms.goals, pms.assists,
       pms.shots_on_target, pms.rating, pms.xg, pms.xa
FROM player_match_stats pms
WHERE pms.player_id = 1000
ORDER BY pms.match_id DESC
LIMIT 50;

-- Q20: Bulk insert performance test (run separately with \timing)
-- INSERT INTO predictions (match_id, model_name, model_version,
--     prob_home, prob_draw, prob_away, predicted_result)
-- SELECT m.id, 'test_model', 'v1',
--        0.45, 0.30, 0.25,
--        CASE WHEN random() < 0.45 THEN 'H'
--             WHEN random() < 0.75 THEN 'D'
--             ELSE 'A' END
-- FROM matches m
-- WHERE m.result IS NOT NULL
-- LIMIT 100000;
