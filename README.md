# Tennis Match Predictor — Backend

A FastAPI service that predicts ATP tennis match outcomes, with a focus on Wimbledon 2026. This is the backend for my [tennis_versus project](https://github.com/carminvuong/tennis_versus), serving predictions to a Next.js frontend deployed on Vercel.

**Frontend repo:** https://github.com/carminvuong/tennis_versus

---

## Overview

This service exposes a trained ML model that predicts the probability of a player winning an ATP match. The model was trained on 2017–2026 ATP match data and evaluated on 2026 matches (out-of-sample), achieving **~63.2% accuracy on Wimbledon 2026** and **~70.2% accuracy on general matches.**

## Tech Stack

- **Framework:** FastAPI (Python)
- **Model:** XGBoost
- **Data:** [TennisMyLife dataset (2017–2026)](https://stats.tennismylife.org/tennis-match-database)
- **Deployment:** Railway

## API Endpoints

Deployed on Railway: https://tennisbackend-production.up.railway.app/*

### `GET /all_players`

Returns the list of players available for prediction (used to populate the frontend's searchable dropdowns).

**Response:**
```json
[
  { "players": [...] }
]
```

### `POST /predict`

Predicts the outcome of a match between two players.

**Request body:**
```json
{
  "player_a": "player_id_1",
  "player_b": "player_id_2",
  "surface": "grass"
}
```

**Response:**
```json
{
  "player_a_win_probability": 0.63,
  "player_b_win_probability": 0.37
}
```

## Model Details

**Features used:**
- Age
- Recent form
- Surface-specific form
- Break point pressure
- Overall Elo
- Surface-specific Elo
- Elo differential (player A vs player B)
- Surface one-hot encoding

**Key modeling decisions:**
- Player data is stored in a neutral `player_a` / `player_b` format rather than `winner`/`loser` to avoid data leakage.
- Elo ratings are computed via a single chronological pass over match history to prevent lookahead bias.
- `rank_diff` and `h2h_winrate` were tested and dropped after empirically hurting accuracy.
- Elo ratings were the single largest driver of accuracy (~69–70% on their own); switching from logistic regression to XGBoost produced negligible additional gains, indicating the bottleneck was feature quality, not model architecture.

## Project Structure

```
.
├── main.py                # FastAPI app, endpoint definitions
├── model/                 # Serialized trained model artifacts
├── data/                  # Parquet handoffs from the training pipeline
├── requirements.txt
└── README.md
```

## Future Improvements

- Per-surface and per-round accuracy breakdowns
- Accounting for holding / breaking serve rates
- Feature contribution breakdown per prediction for interpretability
