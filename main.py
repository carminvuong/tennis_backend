from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from xgboost import XGBClassifier
import pandas as pd
import os

from db import get_all_players, get_player_career_range, get_player_elo_history, get_player_snapshot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'tennis_model_xgb.json')

print(f"Looking for model at: {MODEL_PATH}")

app = FastAPI()

# Allow requests from the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://tennis-versus.vercel.app"], # will be default port for Next.js
    allow_methods=["*"],
    allow_headers=["*"],
)

model = XGBClassifier()
model.load_model(MODEL_PATH)
print("Model loaded successfully")

@app.get("/all_players")
def all_players():
    return {"players": get_all_players()} # already sorted (ORDER BY player_name)

@app.get("/player_snapshot")
def player_snapshot(player_name: str, as_of: date):
    snapshot = get_player_snapshot(player_name, as_of)

    # if DNE, return a 404
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"No stats found for {player_name} on or before {as_of}",
        )

    snapshot_date = snapshot.pop('match_date')
    return {
        "player_name": player_name,
        "as_of_requested": as_of,
        "last_played": snapshot_date,
        **snapshot,
    }

@app.get("/player_career_range")
def player_career_range(player_name: str):
    career_range = get_player_career_range(player_name)
    if career_range is None:
        raise HTTPException(
            status_code=404,
            detail=f"No career range found for {player_name}",
        )

    return {"player_name": player_name, **career_range}

@app.get("/player_elo_history")
def player_elo_history(player_name: str):
    history = get_player_elo_history(player_name)
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No Elo history found for {player_name}",
        )

    return {"player_name": player_name, "history": history}

class PredictionRequest(BaseModel): # data and type validation
    player_a: str
    date_a:   date | None = None
    player_b: str
    date_b:   date | None = None
    surface:  str

    @field_validator('surface')
    @classmethod
    def validate_surface(cls, v):
        if v.lower() not in ('hard', 'clay', 'grass'):
            raise ValueError("surface must be one of 'hard', 'clay', 'grass'")
        return v

def resolve_player(player_name, as_of):
    # as_of omitted -> "as of today", so a player who hasn't played in years
    # (retired or otherwise) surfaces a stale last_played instead of silently
    # returning frozen stats with no indication of how old they are
    if as_of is None:
        as_of = date.today()

    snapshot = get_player_snapshot(player_name, as_of)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"No stats found for {player_name} on or before {as_of}",
        )
    last_played = snapshot.pop('match_date')
    return snapshot, last_played

@app.post("/predict")
def predict(req: PredictionRequest):
    if req.player_a == req.player_b:
        raise HTTPException(status_code=400, detail="player_a and player_b must be different")

    stats_a, last_played_a = resolve_player(req.player_a, req.date_a)
    stats_b, last_played_b = resolve_player(req.player_b, req.date_b)

    surface = req.surface.lower()

    features = pd.DataFrame([{
        'age_a':            stats_a['age'],
        'age_b':            stats_b['age'],
        'form_a':           stats_a['recent_form'],
        'form_b':           stats_b['recent_form'],
        'surface_form_a':   stats_a[f'{surface}_form'],
        'surface_form_b':   stats_b[f'{surface}_form'],
        'bp_pressure_a':    stats_a['bp_pressure'],
        'bp_pressure_b':    stats_b['bp_pressure'],
        'elo_a':            stats_a['elo'],
        'elo_b':            stats_b['elo'],
        'surface_elo_a':    stats_a[f'{surface}_elo'],
        'surface_elo_b':    stats_b[f'{surface}_elo'],
        'elo_diff':         stats_a['elo'] - stats_b['elo'],
        'surface_elo_diff': stats_a[f'{surface}_elo'] - stats_b[f'{surface}_elo'],
        'surface_Clay':     1 if surface == 'clay'  else 0,
        'surface_Grass':    1 if surface == 'grass' else 0,
        'surface_Hard':     1 if surface == 'hard'  else 0,
    }])

    # returns prob of player A winning
    prob = model.predict_proba(features)[0][1]

    return {
        "player_a": req.player_a,
        "player_b": req.player_b,
        "surface": surface,
        "prob_a": round(float(prob), 3),
        "prob_b": round(float(1-prob), 3),
        "last_played_a": last_played_a,
        "last_played_b": last_played_b,
    }