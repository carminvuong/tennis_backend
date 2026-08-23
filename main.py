from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBClassifier
import pandas as pd
import os

from db import get_player_snapshot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'tennis_model_xgb.json')
PLAYER_STATS_PATH = os.path.join(BASE_DIR, 'model', 'data', 'player_stats.csv')
# ^ headers: player,rank,age,elo,recent_form,bp_pressure,grass_elo,grass_form,clay_elo,clay_form,hard_elo,hard_form

print(f"Looking for model at: {MODEL_PATH}")
print(f"Looking for player stats at: {PLAYER_STATS_PATH}")
print(f"Files in BASE_DIR: {os.listdir(BASE_DIR)}")

app = FastAPI()

# Allow requests from the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://tennis-versus.vercel.app"], # will be default port for Next.js
    allow_methods=["*"],
    allow_headers=["*"],
)

# load players data + model
players = pd.read_csv(PLAYER_STATS_PATH, index_col='player') # the index will be the player name
model = XGBClassifier()
model.load_model(MODEL_PATH)    
print("Both files loaded successfully")

@app.get("/all_players")
def all_players():
    return {"players" : sorted(players.index.tolist())} # a list of player names in alphabetical order

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

class PredictionRequest(BaseModel): # data and type validation
    player_a: str
    date_a:   date | None = None
    player_b: str
    date_b:   date | None = None
    surface:  str

def resolve_player(player_name, as_of):
    # as_of omitted -> current behavior, unchanged: most recent stats from the CSV lookup
    if as_of is None:
        try:
            return players.loc[player_name], None
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Player not found: {player_name}")

    # as_of given -> historical snapshot from player_ratings_history
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

    response = {
        "player_a": req.player_a,
        "player_b": req.player_b,
        "surface": surface,
        "prob_a": round(float(prob), 3),
        "prob_b": round(float(1-prob), 3)
    }
    # only present when a historical date was requested for that player -
    # omitting both dates reproduces the exact response shape from before this change
    if last_played_a is not None:
        response["last_played_a"] = last_played_a
    if last_played_b is not None:
        response["last_played_b"] = last_played_b

    return response