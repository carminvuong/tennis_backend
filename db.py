import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def get_all_players():
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT player_name FROM player_ratings_history ORDER BY player_name;")
            rows = cur.fetchall()
    finally:
        conn.close()

    return [row[0] for row in rows]


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


ELO_HISTORY_COLUMNS = ['match_date', 'elo', 'hard_elo', 'clay_elo', 'grass_elo']


def get_player_elo_history(player_name):
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {', '.join(ELO_HISTORY_COLUMNS)}
                FROM player_ratings_history
                WHERE player_name = %s
                ORDER BY match_date ASC;
                """,
                (player_name,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [dict(zip(ELO_HISTORY_COLUMNS, row)) for row in rows]
