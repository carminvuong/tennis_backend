# Tennis Versus — Backend

A FastAPI service that predicts ATP tennis match outcomes, including **cross-era match-ups**, using each player's stats as of any date in their tracked career, not just their current form. This is the backend for [tennis_versus](https://github.com/carminvuong/tennis_versus), a Next.js frontend deployed on Vercel.

**Frontend repo:** https://github.com/carminvuong/tennis_versus

**Live API:** https://tennis-backend-tmoj.onrender.com

---

## Overview

Given two players, a surface, and (optionally) a date for each player, this service predicts a win probability using an XGBoost model fed by each player's Elo rating, recent form, and break-point pressure — resolved as of that specific point in their career via a point-in-time snapshot table in Postgres.

Omitting a date resolves to "as of today," so the same endpoint naturally covers both a same-era prediction (e.g. Sinner vs. Alcaraz right now) and a cross-era one (e.g. peak Federer vs. current Sinner).

## Tech Stack

- **Framework:** FastAPI (Python)
- **Model:** XGBoost, serialized via its native format for a portable, inspectable artifact
- **Database:** Postgres via [Supabase](https://supabase.com) — stores one row per player per match, used for point-in-time snapshot queries
- **Data source:** [TennisMyLife dataset](https://stats.tennismylife.org/tennis-match-database), covering 1991–present (shot-level stats like aces and break points aren't reliably recorded before 1991)
- **Deployment:** Render (backend), Supabase (database)

## API Endpoints

### `GET /all_players`
Returns every player name tracked in the database (sourced from Postgres).

```json
{ "players": ["Aaron Gil Garcia", "Novak Djokovic", "..."] }
```

### `GET /player_snapshot?player_name={name}&as_of={YYYY-MM-DD}`
Returns a player's stats as of the most recent match on or before the given date: Elo (overall + per surface), recent form (overall + per surface), break-point pressure, rank, and age.

```json
{
  "player_name": "Roger Federer",
  "as_of_requested": "2006-07-01",
  "last_played": "2006-06-26",
  "surface": "Grass",
  "rank": 1.0,
  "age": 24.882,
  "elo": 2400.02,
  "recent_form": 0.9,
  "bp_pressure": 5.3,
  "hard_elo": 2447.96, "hard_form": 1.0,
  "clay_elo": 2190.35, "clay_form": 0.9,
  "grass_elo": 2186.91, "grass_form": 1.0
}
```
`404` if the date predates the player's earliest tracked match, or the player isn't tracked at all.

### `GET /player_career_range?player_name={name}`
Returns a player's earliest and latest tracked match dates. Used for bounding date pickers on the frontend.

```json
{ "player_name": "Pete Sampras", "first_match_date": "1991-02-18", "last_match_date": "2002-08-26" }
```

### `GET /player_elo_history?player_name={name}`
Returns a player's full Elo history: one entry per tracked match, chronological, overall + per surface. Powers the `/elo` page's career trajectory chart.

```json
{
  "player_name": "Novak Djokovic",
  "history": [
    { "match_date": "2004-07-19", "elo": 1500.0, "hard_elo": 1500.0, "clay_elo": 1500.0, "grass_elo": 1500.0 },
    "..."
  ]
}
```

### `POST /predict`
Predicts the outcome of a match. `date_a`/`date_b` are optional — a player without a date resolves to their current (most recent) stats.

**Request:**
```json
{
  "player_a": "Roger Federer",
  "date_a": "2006-07-01",
  "player_b": "Jannik Sinner",
  "date_b": "2026-07-01",
  "surface": "grass"
}
```

**Response:**
```json
{
  "player_a": "Roger Federer",
  "player_b": "Jannik Sinner",
  "surface": "grass",
  "prob_a": 0.582,
  "prob_b": 0.418,
  "last_played_a": "2006-06-26",
  "last_played_b": "2026-07-01"
}
```
`last_played_a`/`last_played_b` reflect the actual date each snapshot's stats are from — useful as a staleness signal when a player hasn't competed recently (including retired players, when no date is given).

## Database

A single Postgres table, `player_ratings_history`, holds one row per player per tracked match — their stats immediately before that match, never including its outcome (leakage-safe). A `player_career_range` view derives each player's tracked span from it. Currently: **~195k rows, ~2,570 players, spanning 1991–present.**

The table deliberately does **not** store match outcomes or opponents — that's what makes it safe for point-in-time snapshot queries, but also why model *training* reads from the raw match CSVs directly rather than this table (see below).

## Model Training

Training is a separate, reproducible pipeline — not the notebooks, and not dependent on the Postgres data:

- **`features.py`** — the shared Elo/form/break-point-pressure computation (a single leakage-safe chronological pass), used by both the database backfill and model training, so the two stay consistent.
- **`backfill.py`** — populates `player_ratings_history` from the raw match CSVs in `model/data/`.
- **`train_model.py`** — reproduces the model from scratch: same feature set, same XGBoost hyperparameters, trains on a recent window (2023–2026) with a longer Elo burn-in (2017+) for better-converged ratings, saves via `model.save_model(...)` to `model/tennis_model_xgb.json`.

Run `python train_model.py` to retrain after adding new match data.

**Model features:** age, recent form (overall + surface), break-point pressure, Elo (overall + surface), Elo differential, one-hot surface — 17 features total, unchanged since the original notebook-based version.

**Historical development:** the original feature engineering and model selection (logistic regression → XGBoost, why Elo dominated the other features) is all documented in `model/notebooks/`.

## Project Structure

```
.
├── main.py                # actual FastAPI app, endpoint definitions
├── db.py                  # Postgres queries (snapshot, career range, Elo history, player list)
├── features.py            # Shared Elo/form/bp_pressure computation
├── backfill.py            # Populates player_ratings_history from raw match data
├── train_model.py         # Reproducible model training
├── model/
│   ├── tennis_model_xgb.json   # Trained model (native XGBoost format)
│   ├── data/                   # Raw yearly match CSVs (1991–present) + notebook data files
│   └── notebooks/              # Original exploratory development (feature engineering, model selection)
├── requirements.txt
└── README.md

```

## Future Improvements

- Data from 1991 and earlier wouldn't be a good idea (shot-level stats limit; break points and serve stats aren't recorded), but a live-update mechanism for in-progress tournaments would be a nice next step.
- Feature contribution breakdown per prediction for interpretability.
- Maybe more features --> maybe better model?
