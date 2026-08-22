import os

from psycopg2.extras import execute_values

from db import get_connection
from features import (
    compute_bp_pressure_history,
    compute_elo_history,
    compute_form_history,
    load_and_clean_matches,
    restructure_matches,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'model', 'data')
YEARS = list(range(2009, 2027)) + ['ongoing_tourneys']

INSERT_COLUMNS = [
    'player_name', 'match_date', 'surface', 'rank', 'age', 'elo',
    'recent_form', 'bp_pressure', 'hard_elo', 'hard_form',
    'clay_elo', 'clay_form', 'grass_elo', 'grass_form',
]


def build_history():
    print(f"Loading match data for {YEARS} from {DATA_DIR}...")
    df = load_and_clean_matches(DATA_DIR, YEARS)
    print(f"{len(df)} cleaned matches")

    matches = restructure_matches(df)
    print(f"{len(matches)} player-match rows (2x matches, one row per player per match)")

    print("Computing Elo (overall + per surface)...")
    matches = compute_elo_history(matches)

    print("Computing recent form (overall + per surface)...")
    matches = compute_form_history(matches)

    print("Computing break-point pressure...")
    matches = compute_bp_pressure_history(matches)

    matches = matches.rename(columns={
        'player_a': 'player_name',
        'tourney_date': 'match_date',
        'rank_a': 'rank',
        'age_a': 'age',
    })
    matches['match_date'] = matches['match_date'].dt.date

    return matches[INSERT_COLUMNS]


def insert_history(matches, batch_size=5000):
    rows = list(matches.itertuples(index=False, name=None))
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            columns_sql = ", ".join(INSERT_COLUMNS)
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                execute_values(
                    cur,
                    f"INSERT INTO player_ratings_history ({columns_sql}) VALUES %s",
                    batch,
                )
                print(f"Inserted {min(i + batch_size, len(rows))}/{len(rows)} rows")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    matches = build_history()
    insert_history(matches)
    print("Backfill complete.")
