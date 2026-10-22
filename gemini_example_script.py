import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Path to real CSV directory
DATA_DIR = os.path.join(".", "ac", "hockey_development_data", "hockey_development_data")

def load_and_preprocess_raw_data(data_dir):
    """Loads CSV files and enforces clean numeric data types."""
    print("Loading CSV files...")
    teams = pd.read_csv(os.path.join(data_dir, "Teams.csv"))
    coaches = pd.read_csv(os.path.join(data_dir, "Coaches.csv"))
    scoring = pd.read_csv(os.path.join(data_dir, "Scoring.csv"))
    goalies = pd.read_csv(os.path.join(data_dir, "Goalies.csv"))

    # Convert essential numeric columns safely
    numeric_cols = ['year', 'W', 'L', 'Pts', 'rank', 'G', 'GF', 'GA', 'PPG', 'PPC', 'PIM']
    for col in numeric_cols:
        if col in teams.columns:
            teams[col] = pd.to_numeric(teams[col], errors='coerce').fillna(0)

    for col in ['year', 'GP', 'G', 'A', 'Pts', 'PIM']:
        if col in scoring.columns:
            scoring[col] = pd.to_numeric(scoring[col], errors='coerce').fillna(0)

    for col in ['year', 'GP', 'Min', 'W', 'L', 'GA', 'SA', 'SHO']:
        if col in goalies.columns:
            goalies[col] = pd.to_numeric(goalies[col], errors='coerce').fillna(0)

    return teams, coaches, scoring, goalies

def aggregate_player_and_goalie_features(scoring, goalies):
    """Aggregates player and goalie performance to team-season level (t-1)."""
    # 1. Player Scoring Aggregations per Team per Year
    player_agg = scoring.groupby(['year', 'tmID']).agg(
        total_roster_pts=('Pts', 'sum'),
        total_roster_goals=('G', 'sum'),
        total_roster_pim=('PIM', 'sum'),
        top_scorer_pts=('Pts', 'max')
    ).reset_index()

    # 2. Goalie Performance Aggregations per Team per Year
    goalie_agg = goalies.groupby(['year', 'tmID']).agg(
        total_goalie_wins=('W', 'sum'),
        total_shutouts=('SHO', 'sum'),
        total_ga=('GA', 'sum'),
        total_sa=('SA', 'sum')
    ).reset_index()

    # Derived Goalie Save Percentage
    goalie_agg['team_save_pct'] = np.where(
        goalie_agg['total_sa'] > 0,
        1.0 - (goalie_agg['total_ga'] / goalie_agg['total_sa']),
        0.850 # Default baseline if no shot data
    )

    return player_agg, goalie_agg

def extract_primary_coaches(coaches):
    """Identifies the head coach for each team in each season."""
    # Ensure stint is numeric
    coaches['stint'] = pd.to_numeric(coaches['stint'], errors='coerce').fillna(1)
    
    # Sort by stint to get initial head coach (stint 1) and final head coach
    first_stint = coaches.sort_values(['year', 'tmID', 'stint']).groupby(['year', 'tmID']).first().reset_index()
    last_stint = coaches.sort_values(['year', 'tmID', 'stint']).groupby(['year', 'tmID']).last().reset_index()

    return first_stint[['year', 'tmID', 'coachID']], last_stint[['year', 'tmID', 'coachID']]

def build_temporal_feature_matrix(teams, coaches, scoring, goalies):
    """
    Constructs feature matrix using ONLY season t-1 data to predict season t targets.
    Prevents temporal data leakage.
    """
    player_agg, goalie_agg = aggregate_player_and_goalie_features(scoring, goalies)
    first_coaches, last_coaches = extract_primary_coaches(coaches)

    records = []
    available_years = sorted(teams['year'].unique())

    for t in available_years:
        prev_yr = t - 1
        if prev_yr not in available_years:
            continue # Need prior season to compute features

        curr_teams = teams[teams['year'] == t]
        prev_teams = teams[teams['year'] == prev_yr]

        curr_first_coach = first_coaches[first_coaches['year'] == t]
        prev_last_coach = last_coaches[last_coaches['year'] == prev_yr]

        for _, team_row in curr_teams.iterrows():
            tm = team_row['tmID']

            # Target 1: Regular season final rank at season t
            target_rank = team_row['rank']

            # Target 2: Coaching change (1 if initial coach at season t != last coach at season t-1)
            c_curr = curr_first_coach[curr_first_coach['tmID'] == tm]
            c_prev = prev_last_coach[prev_last_coach['tmID'] == tm]

            if c_curr.empty or c_prev.empty:
                target_coach_change = 0 # Default if coach data missing
            else:
                target_coach_change = int(c_curr.iloc[0]['coachID'] != c_prev.iloc[0]['coachID'])

            # Prior season team features (t-1)
            p_team = prev_teams[prev_teams['tmID'] == tm]
            if p_team.empty:
                continue # Skip expansion or missing prior teams
            p_team = p_team.iloc[0]

            # Prior season roster features (t-1)
            p_player = player_agg[(player_agg['year'] == prev_yr) & (player_agg['tmID'] == tm)]
            p_pts = p_player.iloc[0]['total_roster_pts'] if not p_player.empty else 0
            p_top_pts = p_player.iloc[0]['top_scorer_pts'] if not p_player.empty else 0

            # Prior season goalie features (t-1)
            p_goalie = goalie_agg[(goalie_agg['year'] == prev_yr) & (goalie_agg['tmID'] == tm)]
            p_sv_pct = p_goalie.iloc[0]['team_save_pct'] if not p_goalie.empty else 0.850

            records.append({
                'season': t,
                'tmID': tm,
                # Features from season t-1
                'prev_rank': p_team['rank'],
                'prev_pts': p_team['Pts'],
                'prev_wins': p_team['W'],
                'prev_losses': p_team['L'],
                'prev_gf': p_team['GF'],
                'prev_ga': p_team['GA'],
                'prev_gd': p_team['GF'] - p_team['GA'],
                'prev_roster_pts': p_pts,
                'prev_top_scorer_pts': p_top_pts,
                'prev_save_pct': p_sv_pct,
                # Ground Truth Targets for season t
                'target_rank': target_rank,
                'target_coach_change': target_coach_change
            })

    return pd.DataFrame(records)

def run_evaluation_pipeline():
    teams, coaches, scoring, goalies = load_and_preprocess_raw_data(DATA_DIR)
    
    print("Building temporal feature matrix...")
    df_features = build_temporal_feature_matrix(teams, coaches, scoring, goalies)

    feature_cols = [
        'prev_rank', 'prev_pts', 'prev_wins', 'prev_losses', 
        'prev_gf', 'prev_ga', 'prev_gd', 'prev_roster_pts', 
        'prev_top_scorer_pts', 'prev_save_pct'
    ]

    # Time-Aware Validation Split (e.g., Train on earlier seasons, Test on final available season)
    test_season = df_features['season'].max()
    train_df = df_features[df_features['season'] < test_season]
    test_df = df_features[df_features['season'] == test_season]

    print(f"\n--- Validation Split ---")
    print(f"Train seasons: {train_df['season'].min()} to {train_df['season'].max()} ({len(train_df)} rows)")
    print(f"Test season: {test_season} ({len(test_df)} rows)\n")

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    y_train_rank, y_test_rank = train_df['target_rank'], test_df['target_rank']
    y_train_coach, y_test_coach = train_df['target_coach_change'], test_df['target_coach_change']

    # -----------------------------------------------------------------
    # TASK (a): Predict Regular Season Team Ranking
    # -----------------------------------------------------------------
    print("=== TASK (a): Team Final Rank Prediction ===")
    
    # 1. Naive Baseline: Predict season t rank = season t-1 rank
    dummy_rank_mae = mean_absolute_error(y_test_rank, X_test['prev_rank'])
    print(f"Naive Baseline (Predict prior rank) MAE: {dummy_rank_mae:.3f}")

    # 2. ML Baseline: Ridge Regression
    rank_model = Ridge()
    rank_model.fit(X_train, y_train_rank)
    ml_rank_pred = rank_model.predict(X_test)
    ml_rank_mae = mean_absolute_error(y_test_rank, ml_rank_pred)
    print(f"Ridge Regression Baseline MAE:          {ml_rank_mae:.3f}\n")

    # -----------------------------------------------------------------
    # TASK (b): Predict Coaching Changes
    # -----------------------------------------------------------------
    print("=== TASK (b): Coaching Change Prediction ===")
    
    # 1. Naive Baseline: Always predict 0 (no coach change)
    dummy_coach_pred = np.zeros(len(y_test_coach))
    dummy_f1 = f1_score(y_test_coach, dummy_coach_pred, zero_division=0)
    print(f"Naive Baseline (Predict majority 0) F1-Score: {dummy_f1:.3f}")

    # 2. ML Baseline: Random Forest
    coach_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    coach_model.fit(X_train, y_train_coach)
    ml_coach_pred = coach_model.predict(X_test)
    ml_coach_proba = coach_model.predict_proba(X_test)[:, 1]

    ml_f1 = f1_score(y_test_coach, ml_coach_pred, zero_division=0)
    ml_auc = roc_auc_score(y_test_coach, ml_coach_proba) if len(np.unique(y_test_coach)) > 1 else 0.5
    print(f"Random Forest Classifier F1-Score:           {ml_f1:.3f} | ROC-AUC: {ml_auc:.3f}")

if __name__ == "__main__":
    run_evaluation_pipeline()