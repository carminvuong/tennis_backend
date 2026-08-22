from bisect import bisect_left

import pandas as pd

SURFACES = ['Grass', 'Clay', 'Hard']


def load_and_clean_matches(data_dir, years):
    dfs = [pd.read_csv(f"{data_dir}/{year}.csv") for year in years]
    df = pd.concat(dfs, ignore_index=True)

    df = df[df['surface'].notna()].reset_index(drop=True)
    df = df[df['l_bpSaved'].notna()].reset_index(drop=True)
    df = df.drop(columns=['indoor'], errors='ignore')
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')

    df['winner_rank'] = df['winner_rank'].fillna(2000)
    df['loser_rank'] = df['loser_rank'].fillna(2000)
    df['winner_rank_points'] = df['winner_rank_points'].fillna(0)
    df['loser_rank_points'] = df['loser_rank_points'].fillna(0)

    df = df[df['surface'] != 'Carpet'].reset_index(drop=True)

    df['l_bpFaced'] = pd.to_numeric(df['l_bpFaced'], errors='coerce')
    df['w_bpFaced'] = pd.to_numeric(df['w_bpFaced'], errors='coerce')

    df.loc[(df['loser_name'] == 'Botic Van De Zandschulp') & (df['tourney_name'] == 'Hong Kong'), 'loser_age'] = 30.5
    df.loc[(df['loser_name'] == 'Martin Damm') & (df['loser_age'].isnull()), 'loser_age'] = 22.5
    df = df.dropna(subset=['winner_age', 'loser_age'])

    return df


def restructure_matches(df):
    df_a = pd.DataFrame({
        'tourney_date': df['tourney_date'],
        'surface':      df['surface'],
        'player_a':     df['winner_name'],
        'player_b':     df['loser_name'],
        'rank_a':       df['winner_rank'],
        'age_a':        df['winner_age'],
        'bpFaced_a':    df['w_bpFaced'],
        'label':        1,
    })

    df_b = pd.DataFrame({
        'tourney_date': df['tourney_date'],
        'surface':      df['surface'],
        'player_a':     df['loser_name'],
        'player_b':     df['winner_name'],
        'rank_a':       df['loser_rank'],
        'age_a':        df['loser_age'],
        'bpFaced_a':    df['l_bpFaced'],
        'label':        0,
    })

    result = pd.concat([df_a, df_b], ignore_index=True)
    result = result.sort_values('tourney_date', kind='stable').reset_index(drop=True)
    return result


def compute_elo_history(matches, k=32, default_elo=1500):
    """Mirrors the notebook's compute_elo exactly (02_restructuring_data.ipynb),
    including iterating over the doubled (player_a/player_b) frame — each real
    match applies two sequential rating updates, not one. Replicated deliberately,
    not fixed, so history stays consistent with the already-deployed model and
    player_stats.csv, which were built the same way. The one addition: recording
    ALL THREE surface ratings on every row (the notebook version only records the
    rating for that match's own surface), since a point-in-time snapshot needs
    hard/clay/grass elo simultaneously, not just whichever surface was last played.
    """
    overall_ratings = {}
    surface_ratings = {s: {} for s in SURFACES}

    elo_list = []
    surface_elo_lists = {s: [] for s in SURFACES}

    matches = matches.sort_values('tourney_date', kind='stable').reset_index(drop=True)

    for _, row in matches.iterrows():
        player_a = row['player_a']
        player_b = row['player_b']
        surface = row['surface']

        elo_a = overall_ratings.get(player_a, default_elo)
        elo_b = overall_ratings.get(player_b, default_elo)
        surface_elo_a = surface_ratings[surface].get(player_a, default_elo)
        surface_elo_b = surface_ratings[surface].get(player_b, default_elo)

        elo_list.append(elo_a)
        for s in SURFACES:
            surface_elo_lists[s].append(surface_ratings[s].get(player_a, default_elo))

        overall_expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        surface_expected_a = 1 / (1 + 10 ** ((surface_elo_b - surface_elo_a) / 400))

        actual_a = row['label']
        actual_b = 1 - actual_a

        overall_ratings[player_a] = elo_a + k * (actual_a - overall_expected_a)
        overall_ratings[player_b] = elo_b + k * (actual_b - (1 - overall_expected_a))

        surface_ratings[surface][player_a] = surface_elo_a + k * (actual_a - surface_expected_a)
        surface_ratings[surface][player_b] = surface_elo_b + k * (actual_b - (1 - surface_expected_a))

    matches['elo'] = elo_list
    for s in SURFACES:
        matches[f'{s.lower()}_elo'] = surface_elo_lists[s]

    return matches


def _pointintime_rolling_mean(dates, values, query_dates, last_N, default, min_count=3):
    """For each date in query_dates, returns the mean of the last last_N `values`
    strictly before that date (matching the notebook's `date < row.date` filter,
    including its same-day-ties-excluded behavior), or `default` if fewer than
    min_count prior values exist. dates/values must be sorted by date ascending.
    """
    out = []
    for d in query_dates:
        k = bisect_left(dates, d)
        window = values[max(0, k - last_N):k]
        out.append(window.mean() if len(window) >= min_count else default)
    return out


def compute_form_history(matches, last_N=10):
    """Same definition as the notebook's find_recent_forms/find_recent_forms_surface
    (mean label of a player's last N matches strictly before this row's date, 0.5
    default under 3 prior matches), computed via a per-player point-in-time lookup
    instead of the notebook's per-row full-frame filter. Same output values, far
    faster than the O(n^2) original — necessary here since every row now needs a
    lookup against all 3 surfaces, not just its own.
    """
    matches = matches.sort_values('tourney_date', kind='stable').reset_index(drop=True)

    recent_form = pd.Series(0.5, index=matches.index)
    surface_form = {s.lower(): pd.Series(0.5, index=matches.index) for s in SURFACES}

    for _, group in matches.groupby('player_a'):
        group = group.sort_values('tourney_date')
        idx = group.index.to_numpy()
        dates = group['tourney_date'].to_numpy()
        labels = group['label'].to_numpy()

        recent_form.loc[idx] = _pointintime_rolling_mean(dates, labels, dates, last_N, 0.5)

        for s in SURFACES:
            s_mask = (group['surface'] == s).to_numpy()
            s_dates = dates[s_mask]
            s_labels = labels[s_mask]
            surface_form[s.lower()].loc[idx] = _pointintime_rolling_mean(
                s_dates, s_labels, dates, last_N, 0.5
            )

    matches['recent_form'] = recent_form
    for s in SURFACES:
        matches[f'{s.lower()}_form'] = surface_form[s.lower()]

    return matches


def compute_bp_pressure_history(matches, last_N=10):
    """Same definition as the notebook's bp_pressure (mean break points faced in
    a player's last N matches strictly before this row's date, default 5 under 3
    prior matches), via the same point-in-time lookup as compute_form_history.
    """
    matches = matches.sort_values('tourney_date', kind='stable').reset_index(drop=True)
    bp_pressure = pd.Series(5.0, index=matches.index)

    for _, group in matches.groupby('player_a'):
        group = group.sort_values('tourney_date')
        idx = group.index.to_numpy()
        dates = group['tourney_date'].to_numpy()
        # matches the notebook's `.sum() / len()` (pandas .sum() skips NaN by
        # default but len() still counts the row) — filling with 0 first gives
        # the identical result via a plain mean.
        faced = group['bpFaced_a'].fillna(0).to_numpy()

        bp_pressure.loc[idx] = _pointintime_rolling_mean(dates, faced, dates, last_N, 5.0)

    matches['bp_pressure'] = bp_pressure
    return matches
