from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'tennis_model_xgb_wimbledon.pkl')
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
model = pickle.load(open(MODEL_PATH, "rb")) # rb - read binary, Halle / Queens being the last tourney recorded (before Wimbledon)
print("Both files loaded successfully")

@app.get("/all_players")
def all_players():
    return {"players" : sorted(players.index.tolist())} # a list of player names in alphabetical order 

class PredictionRequest(BaseModel): # data and type validation
    player_a: str
    player_b: str
    surface:  str

@app.post("/predict")
def predict(req: PredictionRequest):
    # find player
    try:
        stats_a = players.loc[req.player_a]
        stats_b = players.loc[req.player_b]
    except KeyError as e: # DNE
        raise HTTPException(status_code=404, detail=f"Player not found: {e}")

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
        "prob_b": round(float(1-prob), 3)
    }