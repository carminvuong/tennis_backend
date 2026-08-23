import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


def get_connection():
    return psycopg2.connect(DATABASE_URL)


SNAPSHOT_COLUMNS = [
    'match_date', 'surface', 'rank', 'age', 'elo', 'recent_form', 'bp_pressure',
    'hard_elo', 'hard_form', 'clay_elo', 'clay_form', 'grass_elo', 'grass_form',
]


def get_player_snapshot(player_name, as_of):
    # connect
    conn = get_connection()

    try:
        with conn.cursor() as cur:

            # find the most recent match before the given date
            cur.execute(
                f"""
                SELECT {', '.join(SNAPSHOT_COLUMNS)}
                FROM player_ratings_history
                WHERE player_name = %s AND match_date <= %s
                ORDER BY match_date DESC
                LIMIT 1;
                """,
                (player_name, as_of),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    # if DNE
    if row is None:
        return None

    return dict(zip(SNAPSHOT_COLUMNS, row))


def get_player_career_range(player_name):
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT first_match_date, last_match_date
                FROM player_career_range
                WHERE player_name = %s;
                """,
                (player_name,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return {'first_match_date': row[0], 'last_match_date': row[1]}
