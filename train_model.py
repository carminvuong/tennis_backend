import os

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from features import (
    build_training_table,
    compute_bp_pressure_history,
    compute_elo_history,
    compute_form_history,
    load_and_clean_matches,
    restructure_matches,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'model', 'data')
MODEL_OUT_PATH = os.path.join(BASE_DIR, 'model', 'tennis_model_xgb.json')

# Matches the range notebook 01 was actually run against (not the full
# 2009-2026 the backfill uses) — kept as-is here to replicate current
# production behavior. Extending this to test the Section-4 open decision
# (a longer Elo burn-in window than the training window) is a deliberate,
# separate change, not something to fold in silently.
YEARS = list(range(2017, 2027)) + ['ongoing_tourneys']

TRAIN_START = '2023-01-01'
TRAIN_END = '2026-01-01'

FEATURES = [
    'age_a', 'age_b', 'form_a', 'form_b', 'surface_form_a', 'surface_form_b',
    'bp_pressure_a', 'bp_pressure_b', 'elo_a', 'elo_b', 'surface_elo_a', 'surface_elo_b',
    'elo_diff', 'surface_elo_diff', 'surface_Clay', 'surface_Grass', 'surface_Hard',
]


def build_dataset():
    print(f"Loading match data for {YEARS} from {DATA_DIR}...")
    df = load_and_clean_matches(DATA_DIR, YEARS)
    print(f"{len(df)} cleaned matches")

    matches = restructure_matches(df)

    print("Computing Elo (overall + per surface)...")
    matches = compute_elo_history(matches)

    print("Computing recent form (overall + per surface)...")
    matches = compute_form_history(matches)

    print("Computing break-point pressure...")
    matches = compute_bp_pressure_history(matches)

    print("Assembling wide (both-players-per-row) training table...")
    data = build_training_table(matches)

    return data


def train(data):
    train_data = data[(data['tourney_date'] >= TRAIN_START) & (data['tourney_date'] < TRAIN_END)]
    test_data = data[data['tourney_date'] >= TRAIN_END]

    X_train, y_train = train_data[FEATURES], train_data['label']
    X_test, y_test = test_data[FEATURES], test_data['label']

    print(f"train: {X_train.shape}, test: {X_test.shape}")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Log loss:  {log_loss(y_test, y_pred_proba):.3f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.3f}")

    return model


if __name__ == "__main__":
    data = build_dataset()
    model = train(data)
    model.save_model(MODEL_OUT_PATH)
    print(f"Model saved to {MODEL_OUT_PATH}")
