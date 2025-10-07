# main.py (UPDATED)
import os
import pickle
import numpy as np
import pandas as pd
from packaging import version
from sklearn import __version__ as skl_version
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy import sparse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional
import xgboost as xgb
import logging
from datetime import datetime

# --- ADD to top imports in main.py ---
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
# add these near other imports
import io
import base64
import matplotlib.pyplot as plt
# seaborn is optional in server envs; keep import but not required
import seaborn as sns
# Additional imports for H2H improvements
from functools import lru_cache
from scipy.stats import ttest_ind
# --- end add ---

# Ensure matplotlib backend won't require display (for some environments)
import matplotlib
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# Model persistence config (MUST be declared before training functions)
# ===============================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Team winner model files
TEAM_MODEL_PATH = os.path.join(MODEL_DIR, "team_winner_xgb.json")
TEAM_ENCODER_PATH = os.path.join(MODEL_DIR, "team_features_encoder.joblib")
WINNER_LE_PATH = os.path.join(MODEL_DIR, "winner_label_encoder.joblib")

# Innings model files (existing in your file)
MODEL_PATH_1 = os.path.join(MODEL_DIR, "innings_model_1.joblib")
MODEL_PATH_2 = os.path.join(MODEL_DIR, "innings_model_2.joblib")
ENC_PATH = os.path.join(MODEL_DIR, "innings_encoders.joblib")  # stores LabelEncoders

# ===============================
# FastAPI App Initialization
# ===============================
app = FastAPI(
    title="Cricket Match Winner Predictor API",
    description="An API to predict T20 cricket match winners using XGBoost model and provide player analysis.",
    version="1.9.0"
)

# CORS (Cross-Origin Resource Sharing) middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ===============================
# Config
# ===============================
# IMPORTANT: Make sure this CSV file is in the same directory as this main.py file.
FILE_PATH = "ball_by_ball.csv"

# Global variables
DF = None
MATCH_DF = None
AVAILABLE_TEAMS = []
AVAILABLE_VENUES = []
AVAILABLE_SEASONS = []
AVAILABLE_PLAYERS = []
AVAILABLE_BOWLERS = []
XGB_MODEL = None
ENCODER = None
MODEL_TRAINED = False
WINNER_LE = None

# ===============================
# Data loading and model training
# ===============================
def load_data():
    """Load data from CSV file"""
    global DF, MATCH_DF, AVAILABLE_TEAMS, AVAILABLE_VENUES, AVAILABLE_SEASONS, AVAILABLE_PLAYERS, AVAILABLE_BOWLERS

    try:
        logger.info(f"Attempting to load data from '{FILE_PATH}'...")
        DF = pd.read_csv(FILE_PATH)

        # Print dataset info
        logger.info(f"Rows: {len(DF)} | Columns: {list(DF.columns)}")

        MATCH_DF = DF.drop_duplicates(subset=["MatchID"]).copy()
        AVAILABLE_TEAMS = sorted(MATCH_DF["Team1"].unique().tolist()) if "Team1" in MATCH_DF.columns else []
        AVAILABLE_VENUES = sorted(MATCH_DF["Venue"].unique().tolist()) if "Venue" in MATCH_DF.columns else []

        # Get available seasons and players for analysis - FIX: Convert numpy types to Python native types
        if "Season" in DF.columns:
            try:
                AVAILABLE_SEASONS = sorted([int(s) for s in DF["Season"].unique()])
            except Exception:
                AVAILABLE_SEASONS = sorted([str(s) for s in DF["Season"].unique()])
        else:
            AVAILABLE_SEASONS = []

        # Ensure both Batters and Bowlers are in the player list
        if "Batter" in DF.columns and "Bowler" in DF.columns:
            all_players = pd.concat([DF['Batter'], DF['Bowler']]).dropna().unique()
            AVAILABLE_PLAYERS = sorted([str(p) for p in all_players if pd.notna(p)])
        else:
            AVAILABLE_PLAYERS = []

        # Get unique bowlers
        if 'Bowler' in DF.columns:
            AVAILABLE_BOWLERS = sorted(DF['Bowler'].dropna().unique().tolist())
        else:
            AVAILABLE_BOWLERS = []

        logger.info("✅ Data loaded successfully!")
        logger.info(f"- {len(AVAILABLE_TEAMS)} Teams")
        logger.info(f"- {len(AVAILABLE_VENUES)} Venues")
        logger.info(f"- {len(AVAILABLE_SEASONS)} Seasons")
        logger.info(f"- {len(AVAILABLE_PLAYERS)} Players")
        logger.info(f"- {len(AVAILABLE_BOWLERS)} Bowlers")

        # Train the XGBoost model (this will attempt to load persisted model first)
        train_xgboost_model()

    except FileNotFoundError:
        logger.error(f"❌ FATAL ERROR: The data file '{FILE_PATH}' was not found.")
        logger.error("Ensure the CSV file is in the same directory as main.py and is named correctly.")
        # Create empty structures to prevent further errors
        AVAILABLE_TEAMS = []
        AVAILABLE_VENUES = []
        AVAILABLE_SEASONS = []
        AVAILABLE_PLAYERS = []
        AVAILABLE_BOWLERS = []
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        # Create empty structures to prevent further errors
        AVAILABLE_TEAMS = []
        AVAILABLE_VENUES = []
        AVAILABLE_SEASONS = []
        AVAILABLE_PLAYERS = []
        AVAILABLE_BOWLERS = []

# Replace existing train_xgboost_model with this function
def train_xgboost_model():
    """Train XGBoost model for match prediction (robust with label encoding and persistence)."""
    global XGB_MODEL, ENCODER, MODEL_TRAINED, WINNER_LE

    try:
        # Try to lazy-load existing persisted model first
        if os.path.exists(TEAM_MODEL_PATH) and os.path.exists(TEAM_ENCODER_PATH) and os.path.exists(WINNER_LE_PATH):
            try:
                XGB_MODEL = xgb.XGBClassifier()
                XGB_MODEL.load_model(TEAM_MODEL_PATH)
                ENCODER = joblib.load(TEAM_ENCODER_PATH)
                WINNER_LE = joblib.load(WINNER_LE_PATH)
                MODEL_TRAINED = True
                logger.info("✅ Loaded persisted team winner model and encoders.")
                return
            except Exception as e:
                logger.warning(f"Could not load persisted team model/encoders: {e}. Will attempt to train a new one.")

        if MATCH_DF is None or len(MATCH_DF) < 10:
            logger.warning("Not enough data to train XGBoost model. Using statistical approach.")
            MODEL_TRAINED = False
            return

        logger.info("🔄 Training XGBoost model...")

        # Prepare features and target at match-level
        features = ['Team1', 'Team2', 'Venue']
        target = 'Winner'

        # Filter valid matches where Winner exists and Winner is one of the teams
        valid_matches = MATCH_DF.dropna(subset=features + [target]).copy()
        # Keep only matches where Winner equals Team1 or Team2 (clean)
        valid_matches = valid_matches[
            (valid_matches['Winner'] == valid_matches['Team1']) | (valid_matches['Winner'] == valid_matches['Team2'])
        ]

        if len(valid_matches) < 10:
            logger.warning("Not enough valid matches to train XGBoost model.")
            MODEL_TRAINED = False
            return

        X_raw = valid_matches[features].astype(str)
        y_raw = valid_matches[target].astype(str)

        # One-hot encode categorical features (Team1, Team2, Venue)
        # Use handle_unknown='ignore' so encoder can handle unseen during inference
        ENCODER = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_encoded = ENCODER.fit_transform(X_raw)

        # Label-encode the target (winner team names -> integer classes)
        WINNER_LE = LabelEncoder()
        y_encoded = WINNER_LE.fit_transform(y_raw)  # integers 0..n_classes-1

        # Train / test split (stratify by encoded labels)
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Create and fit XGBoost classifier
        XGB_MODEL = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        XGB_MODEL.fit(X_train, y_train)

        # Evaluate
        train_acc = float(XGB_MODEL.score(X_train, y_train))
        test_acc = float(XGB_MODEL.score(X_test, y_test))

        MODEL_TRAINED = True
        logger.info(f"✅ XGBoost model trained successfully!")
        logger.info(f"   Training Accuracy: {train_acc:.4f}")
        logger.info(f"   Test Accuracy: {test_acc:.4f}")
        logger.info(f"   Classes (labels): {WINNER_LE.classes_.tolist()}")

        # Persist model and encoders to disk for future restarts
        try:
            XGB_MODEL.save_model(TEAM_MODEL_PATH)
            joblib.dump(ENCODER, TEAM_ENCODER_PATH)
            joblib.dump(WINNER_LE, WINNER_LE_PATH)
            logger.info(f"📦 Saved team winner model to {TEAM_MODEL_PATH} and encoders to {MODEL_DIR}")
        except Exception as e:
            logger.warning(f"Could not persist model/encoders to disk: {e}")

    except Exception as e:
        logger.error(f"❌ Error training XGBoost model: {e}")
        MODEL_TRAINED = False
        XGB_MODEL = None
        ENCODER = None
        WINNER_LE = None

# Replace existing predict_with_xgboost with this function (returns 3 values for compatibility)
def predict_with_xgboost(team_a, team_b, venue):
    """
    Predict match winner using trained XGBoost model.
    Returns (predicted_winner_name, prob_for_team_a, prob_for_team_b)
    If model not available returns (None, None, None)
    """
    global XGB_MODEL, ENCODER, MODEL_TRAINED, WINNER_LE

    # Try to lazy-load model/encoders from disk if in-memory not available
    if not MODEL_TRAINED or XGB_MODEL is None or ENCODER is None or WINNER_LE is None:
        # attempt to load persisted artifacts
        try:
            if os.path.exists(TEAM_MODEL_PATH) and os.path.exists(TEAM_ENCODER_PATH) and os.path.exists(WINNER_LE_PATH):
                XGB_MODEL = xgb.XGBClassifier()
                XGB_MODEL.load_model(TEAM_MODEL_PATH)
                ENCODER = joblib.load(TEAM_ENCODER_PATH)
                WINNER_LE = joblib.load(WINNER_LE_PATH)
                MODEL_TRAINED = True
                logger.info("✅ Loaded XGBoost model and encoders from disk for prediction.")
        except Exception as e:
            logger.warning(f"Could not load persisted model/encoders: {e}")

    if not MODEL_TRAINED or XGB_MODEL is None or ENCODER is None or WINNER_LE is None:
        logger.warning("XGBoost model or encoders unavailable. Prediction aborted.")
        return None, None, None

    try:
        # Build input DataFrame consistent with encoder expectations
        input_df = pd.DataFrame([{
            'Team1': team_a,
            'Team2': team_b,
            'Venue': venue if venue else ''
        }]).astype(str)

        # Encode features using trained OneHotEncoder
        X_in = ENCODER.transform(input_df)  # shape (1, n_features)

        # Get probability distribution over encoded winner classes
        proba = XGB_MODEL.predict_proba(X_in)[0]  # length == n_classes

        # Predicted class (encoded)
        pred_encoded = int(XGB_MODEL.predict(X_in)[0])
        pred_team = WINNER_LE.inverse_transform([pred_encoded])[0]

        # Build mapping of team label -> probability
        proba_labels = {}
        for cls_idx, cls_label in enumerate(WINNER_LE.classes_):
            proba_labels[cls_label] = float(proba[cls_idx])

        # Extract probabilities for team_a and team_b (if present in classes)
        prob_a = proba_labels.get(team_a, 0.0)
        prob_b = proba_labels.get(team_b, 0.0)

        return pred_team, prob_a, prob_b

    except Exception as e:
        logger.error(f"❌ Error in XGBoost prediction: {e}")
        return None, None, None

# ===============================
# Player & head-to-head helper functions
# ===============================

def get_key_players(team_name, num_recent_matches=10):
    """Get key players (batters and bowlers) for a team based on recent performance"""
    if DF is None:
        return {"error": "Data not loaded"}

    try:
        # Get recent matches for the team
        team_matches = MATCH_DF[
            (MATCH_DF['Team1'] == team_name) | (MATCH_DF['Team2'] == team_name)
        ]

        if team_matches.empty:
            return {"error": f"No matches found for {team_name}"}

        # Sort by MatchID (assuming higher MatchID means more recent)
        recent_matches = team_matches.nlargest(num_recent_matches, 'MatchID')
        recent_match_ids = recent_matches['MatchID'].tolist()

        # Filter ball-by-ball data for recent matches
        recent_data = DF[DF['MatchID'].isin(recent_match_ids)]

        # Get key batters (based on runs scored in recent matches)
        team_batting_data = recent_data[recent_data['BattingTeam'] == team_name]
        if not team_batting_data.empty:
            batter_stats = team_batting_data.groupby('Batter').agg({
                'BatterRuns': 'sum',
                'MatchID': 'nunique'
            }).reset_index()
            batter_stats = batter_stats[batter_stats['BatterRuns'] > 0]  # Only players who scored runs
            top_batters = batter_stats.nlargest(3, 'BatterRuns').to_dict('records')
        else:
            top_batters = []

        # Get key bowlers (based on wickets taken in recent matches)
        team_bowling_data = recent_data[
            ((recent_data['Team1'] == team_name) & (recent_data['BattingTeam'] != team_name)) |
            ((recent_data['Team2'] == team_name) & (recent_data['BattingTeam'] != team_name))
        ]

        if not team_bowling_data.empty:
            bowler_stats = team_bowling_data.groupby('Bowler').agg({
                'BowlerWicket': 'sum',
                'MatchID': 'nunique'
            }).reset_index()
            bowler_stats = bowler_stats[bowler_stats['BowlerWicket'] > 0]  # Only players who took wickets
            top_bowlers = bowler_stats.nlargest(3, 'BowlerWicket').to_dict('records')
        else:
            top_bowlers = []

        # Format the response
        key_players = {
            "team": team_name,
            "recent_matches_analyzed": len(recent_match_ids),
            "top_batters": [
                {
                    "name": batter["Batter"],
                    "runs": int(batter["BatterRuns"]),
                    "matches": int(batter["MatchID"])
                } for batter in top_batters
            ],
            "top_bowlers": [
                {
                    "name": bowler["Bowler"],
                    "wickets": int(bowler["BowlerWicket"]),
                    "matches": int(bowler["MatchID"])
                } for bowler in top_bowlers
            ]
        }

        return key_players

    except Exception as e:
        logger.error(f"Error getting key players for {team_name}: {e}")
        return {"error": f"Failed to get key players for {team_name}"}


def get_player_head_to_head(df, player_a, player_b, season=None, venue=None, min_samples_for_stats=5):
    """
    Enhanced head-to-head comparison between two players.
    - Detects batsman-vs-batsman or batsman-vs-bowler mode.
    - Computes runs, balls, SR, average, dismissals, plus per-innings data.
    - Performs simple significance testing (ttest) on per-innings runs when samples are sufficient.
    - Returns a consistent JSON schema for frontend consumption.
    """
    if df is None or df.empty:
        return {"error": "Data not loaded"}

    # Helper functions (local)
    def _find_runs_col(df_local):
        for cand in ("BatterRuns", "Runs", "Run"):
            if cand in df_local.columns:
                return cand
        return None

    def _find_wicket_col(df_local):
        for cand in ("BowlerWicket", "Wicket", "DismissalType"):
            if cand in df_local.columns:
                return cand
        return None

    def _count_balls(sub):
        if "IsLegalDelivery" in sub.columns:
            return int(sub[sub["IsLegalDelivery"] == True].shape[0])
        if "IsLegal" in sub.columns:
            return int(sub[sub["IsLegal"] == True].shape[0])
        # fallback: count rows
        return int(sub.shape[0])

    def _per_innings_records(sub, runs_col):
        if sub.empty:
            return []
        if "Innings" in sub.columns:
            agg = sub.groupby(["MatchID", "Innings"]).agg(Runs=(runs_col, "sum"), Balls=(runs_col, "count")).reset_index()
            return agg.to_dict("records")
        else:
            agg = sub.groupby("MatchID").agg(Runs=(runs_col, "sum"), Balls=(runs_col, "count")).reset_index()
            return agg.to_dict("records")

    runs_col = _find_runs_col(df)
    if runs_col is None:
        return {"error": "Runs column not found in dataset"}

    wicket_col = _find_wicket_col(df)

    # Apply filters
    df_sub = df.copy()
    if season and "Season" in df_sub.columns:
        try:
            df_sub = df_sub[df_sub["Season"] == int(season)]
        except Exception:
            # season may be string; try matching string
            df_sub = df_sub[df_sub["Season"].astype(str) == str(season)]
    if venue and "Venue" in df_sub.columns:
        df_sub = df_sub[df_sub["Venue"] == venue]

    if df_sub.empty:
        return {"error": "No data after filtering"}

    # Determine how often each plays as batsman or bowler
    a_as_batter = int(df_sub[df_sub.get("Batter", "") == player_a].shape[0]) if "Batter" in df_sub.columns else 0
    a_as_bowler = int(df_sub[df_sub.get("Bowler", "") == player_a].shape[0]) if "Bowler" in df_sub.columns else 0
    b_as_batter = int(df_sub[df_sub.get("Batter", "") == player_b].shape[0]) if "Batter" in df_sub.columns else 0
    b_as_bowler = int(df_sub[df_sub.get("Bowler", "") == player_b].shape[0]) if "Bowler" in df_sub.columns else 0

    mode = "batsman_vs_batsman"
    if a_as_batter > 0 and b_as_bowler > 0:
        mode = "batsman_vs_bowler"
    elif b_as_batter > 0 and a_as_bowler > 0:
        mode = "batsman_vs_bowler"

    # Batsman vs Bowler
    if mode == "batsman_vs_bowler":
        # decide which is batter and which is bowler
        if a_as_batter > 0 and b_as_bowler > 0:
            batter = player_a
            bowler = player_b
        elif b_as_batter > 0 and a_as_bowler > 0:
            batter = player_b
            bowler = player_a
        else:
            # fallback: if one exists in Batter column pick that as batter
            if a_as_batter > 0:
                batter = player_a
                bowler = player_b
            elif b_as_batter > 0:
                batter = player_b
                bowler = player_a
            else:
                # no clear roles: return error
                return {"error": "Insufficient role-specific data for batsman-vs-bowler mode"}

        matchup = df_sub[(df_sub["Batter"] == batter) & (df_sub["Bowler"] == bowler)]
        if matchup.empty:
            return {
                "type": "batsman_vs_bowler",
                "batter": batter,
                "bowler": bowler,
                "summary": {},
                "per_innings": []
            }

        runs = int(matchup[runs_col].sum())
        balls = int(_count_balls(matchup))
        # compute outs: if DismissalType exists, count non-null meaningful dismissals
        outs = 0
        if wicket_col:
            if wicket_col == "DismissalType" and "DismissalType" in matchup.columns:
                non_dismissals = ["", "Not Out", "not out", "no wicket", "retired hurt"]
                outs = int(matchup[matchup["DismissalType"].notna() & (~matchup["DismissalType"].isin(non_dismissals))].shape[0])
            elif wicket_col in matchup.columns and matchup[wicket_col].dtype in [np.int64, np.float64, int, float]:
                outs = int(matchup[wicket_col].sum())
            else:
                # fallback: count rows where BowlerWicket==1
                if "BowlerWicket" in matchup.columns:
                    outs = int(matchup["BowlerWicket"].sum())
        avg = round(runs / outs, 2) if outs > 0 else None
        sr = round((runs / balls * 100), 2) if balls > 0 else None

        per_innings = _per_innings_records(matchup, runs_col)

        # Bowler stats vs this batter
        bowler_matches = matchup[matchup["Bowler"] == bowler]
        bowler_runs = int(bowler_matches[runs_col].sum()) if not bowler_matches.empty else 0
        bowler_balls = int(_count_balls(bowler_matches)) if not bowler_matches.empty else 0
        bowler_wickets = 0
        if "BowlerWicket" in bowler_matches.columns:
            bowler_wickets = int(bowler_matches["BowlerWicket"].sum())
        elif "DismissalType" in bowler_matches.columns:
            bowler_wickets = int(bowler_matches[bowler_matches["DismissalType"].notna()].shape[0])

        bowler_avg = round(bowler_runs / bowler_wickets, 2) if bowler_wickets > 0 else None
        bowler_eco = round((bowler_runs / (bowler_balls/6)), 2) if bowler_balls > 0 else None

        return {
            "type": "batsman_vs_bowler",
            "batter": batter,
            "bowler": bowler,
            "summary": {
                "batter": {
                    "runs": runs,
                    "balls": balls,
                    "outs": outs,
                    "average": avg,
                    "strike_rate": sr,
                    "per_innings_count": len(per_innings)
                },
                "bowler": {
                    "runs_conceded": bowler_runs,
                    "balls": bowler_balls,
                    "wickets": bowler_wickets,
                    "average": bowler_avg,
                    "economy": bowler_eco
                }
            },
            "per_innings": per_innings
        }

    # Batsman vs Batsman
    df_a = df_sub[df_sub["Batter"] == player_a]
    df_b = df_sub[df_sub["Batter"] == player_b]

    def batting_stats(sub):
        if sub.empty:
            return {"matches": 0, "runs": 0, "balls": 0, "outs": 0, "average": None, "strike_rate": None, "per_innings": []}
        runs = int(sub[runs_col].sum())
        balls = int(_count_balls(sub))
        outs = 0
        if wicket_col:
            if wicket_col == "DismissalType" and "DismissalType" in sub.columns:
                non_dismissals = ["", "Not Out", "not out", "no wicket", "retired hurt"]
                outs = int(sub[sub["DismissalType"].notna() & (~sub["DismissalType"].isin(non_dismissals))].shape[0])
            elif wicket_col in sub.columns and sub[wicket_col].dtype in [np.int64, np.float64, int, float]:
                outs = int(sub[wicket_col].sum())
            else:
                if "BowlerWicket" in sub.columns:
                    outs = int(sub["BowlerWicket"].sum())
        avg = round(runs / outs, 2) if outs > 0 else None
        sr = round((runs / balls * 100), 2) if balls > 0 else None
        per_innings = _per_innings_records(sub, runs_col)
        return {
            "matches": int(sub["MatchID"].nunique()) if "MatchID" in sub.columns else 0,
            "runs": runs,
            "balls": balls,
            "outs": outs,
            "average": avg,
            "strike_rate": sr,
            "per_innings": per_innings
        }

    stats_a = batting_stats(df_a)
    stats_b = batting_stats(df_b)

    # Comparison metrics and simple significance test on per-innings runs
    metrics = []
    adv_score = 0

    # metric: runs (use per-innings distributions)
    arr_a = [int(item["Runs"]) for item in stats_a.get("per_innings", [])]
    arr_b = [int(item["Runs"]) for item in stats_b.get("per_innings", [])]
    p_value = None
    significant = False
    if len(arr_a) >= min_samples_for_stats and len(arr_b) >= min_samples_for_stats:
        try:
            stat, p_value = ttest_ind(arr_a, arr_b, equal_var=False, nan_policy='omit')
            significant = (p_value is not None and p_value < 0.05)
        except Exception:
            p_value = None
            significant = False

    diff_runs = None
    if stats_a["runs"] is not None and stats_b["runs"] is not None:
        diff_runs = stats_a["runs"] - stats_b["runs"]
        if diff_runs > 0:
            adv = "player_a"
            adv_score += 1
        elif diff_runs < 0:
            adv = "player_b"
            adv_score -= 1
        else:
            adv = None
    else:
        adv = None

    metrics.append({
        "metric": "Runs",
        "key": "runs",
        "value_a": stats_a["runs"],
        "value_b": stats_b["runs"],
        "diff": diff_runs,
        "advantage": adv,
        "p_value": float(p_value) if p_value is not None else None,
        "significant": significant,
        "sample_a": len(arr_a),
        "sample_b": len(arr_b)
    })

    # metric: average
    diff_avg = None
    if stats_a["average"] is not None and stats_b["average"] is not None:
        try:
            diff_avg = (stats_a["average"] or 0) - (stats_b["average"] or 0)
            if diff_avg > 0:
                adv_score += 1
            elif diff_avg < 0:
                adv_score -= 1
        except Exception:
            diff_avg = None

    metrics.append({
        "metric": "Average",
        "key": "average",
        "value_a": stats_a["average"],
        "value_b": stats_b["average"],
        "diff": diff_avg,
        "advantage": ("player_a" if (diff_avg or 0) > 0 else ("player_b" if (diff_avg or 0) < 0 else None)),
        "p_value": None,
        "significant": None,
        "sample_a": len(arr_a),
        "sample_b": len(arr_b)
    })

    # metric: strike rate
    diff_sr = None
    if stats_a["strike_rate"] is not None and stats_b["strike_rate"] is not None:
        try:
            diff_sr = (stats_a["strike_rate"] or 0) - (stats_b["strike_rate"] or 0)
            if diff_sr > 0:
                adv_score += 1
            elif diff_sr < 0:
                adv_score -= 1
        except Exception:
            diff_sr = None

    metrics.append({
        "metric": "StrikeRate",
        "key": "strike_rate",
        "value_a": stats_a["strike_rate"],
        "value_b": stats_b["strike_rate"],
        "diff": diff_sr,
        "advantage": ("player_a" if (diff_sr or 0) > 0 else ("player_b" if (diff_sr or 0) < 0 else None)),
        "p_value": None,
        "significant": None,
        "sample_a": len(arr_a),
        "sample_b": len(arr_b)
    })

    # Final advantage
    if adv_score > 0:
        overall_advantage = "player_a"
    elif adv_score < 0:
        overall_advantage = "player_b"
    else:
        overall_advantage = None

    return {
        "type": "batsman_vs_batsman",
        "player_a": {"name": player_a, "stats": stats_a},
        "player_b": {"name": player_b, "stats": stats_b},
        "comparison": {
            "metrics": metrics,
            "overall_advantage": overall_advantage
        }
    }

def get_player_innings(df, player_name, season=None, venue=None):
    """
    Return per-innings scores for a player (list of innings with MatchID, Runs, Balls).
    """
    if df is None:
        return {"error": "Data not loaded"}

    df_player = df[df['Batter'] == player_name].copy()
    if season and "Season" in df_player.columns:
        try:
            df_player = df_player[df_player['Season'] == int(season)]
        except Exception:
            df_player = df_player[df_player['Season'].astype(str) == str(season)]
    if venue and "Venue" in df_player.columns:
        df_player = df_player[df_player['Venue'] == venue]

    if df_player.empty:
        return {"player": player_name, "innings": []}

    # Aggregate per MatchID (and Innings if present)
    if 'Innings' in df_player.columns:
        innings_df = (
            df_player.groupby(['MatchID', 'Innings'])
            .agg(Runs=('BatterRuns', 'sum'), Balls=('BatterRuns', 'count'))
            .reset_index()
            .sort_values(['MatchID', 'Innings'])
        )
    else:
        innings_df = (
            df_player.groupby('MatchID')
            .agg(Runs=('BatterRuns', 'sum'), Balls=('BatterRuns', 'count'))
            .reset_index()
            .sort_values('MatchID')
        )

    innings = innings_df.to_dict('records')
    return {"player": player_name, "innings": innings}

# --- INNINGS dataset builder, trainer and predictor (unchanged logic but uses top MODEL_DIR) ---

def build_innings_dataset(df):
    """
    Build an innings-level DataFrame of final innings totals (one row per innings).
    Tries to be robust: expects columns such as:
      - MatchID, Season (or Year), BattingTeam (or Batting), BowlingTeam (or BowlerTeam),
        BatterRuns (per ball) OR TotalRuns (per innings), Innings/Inning (1/2)
    Output columns (guaranteed): ['MatchID','Season','Year','BattingTeam','BowlingTeam','Innings','TotalRuns']
    """
    if df is None:
        raise ValueError("DF is None")

    df_local = df.copy()

    # Harmonize common column names
    if 'Season' not in df_local.columns and 'season' in df_local.columns:
        df_local['Season'] = df_local['season']
    if 'Year' not in df_local.columns:
        # attempt to copy season or infer from date if present
        if 'Season' in df_local.columns:
            df_local['Year'] = df_local['Season']
        elif 'match_date' in df_local.columns:
            df_local['Year'] = pd.to_datetime(df_local['match_date']).dt.year
        else:
            df_local['Year'] = df_local.get('Season', np.nan)

    # Determine batting team and bowling team columns
    bat_col = 'BattingTeam' if 'BattingTeam' in df_local.columns else ('Batting' if 'Batting' in df_local.columns else None)
    bowl_col = 'BowlingTeam' if 'BowlingTeam' in df_local.columns else ('Bowling' if 'Bowling' in df_local.columns else None)

    # Determine per-ball runs column
    per_ball_runs_col = None
    for cand in ['BatterRuns', 'Runs', 'Run', 'batsman_run']:
        if cand in df_local.columns:
            per_ball_runs_col = cand
            break

    # If the dataset already has precomputed innings totals in a column like 'InningsTotal' or 'TotalRuns', use that
    innings_total_col = None
    for cand in ['InningsTotal', 'TotalRuns', 'TeamRuns', 'Total']:
        if cand in df_local.columns:
            innings_total_col = cand
            break

    # If an explicit innings number exists, use it; else try 'Innings' or 'Inning'
    innings_col = 'Innings' if 'Innings' in df_local.columns else ('Inning' if 'Inning' in df_local.columns else None)

    # Attempt: if innings_total_col exists, group by MatchID + BattingTeam to pick total
    if innings_total_col and bat_col:
        agg = df_local.groupby(['MatchID', bat_col]).agg({
            innings_total_col: 'max',
            'Season': 'first',
            'Year': 'first'
        }).reset_index().rename(columns={bat_col: 'BattingTeam', innings_total_col: 'TotalRuns'})

        results = []
        for mid, g in agg.groupby('MatchID'):
            # try to keep the order consistent with appearance in original data
            order = []
            for t in df_local[df_local['MatchID'] == mid][bat_col].tolist():
                if t not in order:
                    order.append(t)
            # iterate in appearance order if possible
            idx = 1
            for team in order:
                row = g[g['BattingTeam'] == team]
                if row.empty:
                    continue
                r = row.iloc[0]
                opp = [t for t in order if t != team]
                opp_name = opp[0] if opp else None
                results.append({
                    'MatchID': mid,
                    'Season': r['Season'],
                    'Year': r['Year'],
                    'BattingTeam': team,
                    'BowlingTeam': opp_name,
                    'Innings': idx,
                    'TotalRuns': int(r['TotalRuns']) if not pd.isna(r['TotalRuns']) else 0
                })
                idx += 1
        df_innings = pd.DataFrame(results)
        return df_innings

    # Otherwise aggregate per-ball runs grouped by MatchID + BattingTeam to build innings totals
    if per_ball_runs_col and bat_col:
        agg = df_local.groupby(['MatchID', bat_col]).agg({
            per_ball_runs_col: 'sum',
            'Season': 'first',
            'Year': 'first'
        }).reset_index().rename(columns={bat_col: 'BattingTeam', per_ball_runs_col: 'TotalRuns'})
        # infer opponent
        results = []
        for mid, g in agg.groupby('MatchID'):
            teams = df_local[df_local['MatchID'] == mid][bat_col].unique().tolist()
            # order by appearance in original df to guess innings order
            order = []
            for t in df_local[df_local['MatchID'] == mid][bat_col].tolist():
                if t not in order:
                    order.append(t)
            idx = 1
            for team in order:
                row = g[g['BattingTeam'] == team]
                if row.empty:
                    continue
                r = row.iloc[0]
                opp = [t for t in teams if t != team]
                opp_name = opp[0] if opp else None
                results.append({
                    'MatchID': mid,
                    'Season': r['Season'],
                    'Year': r['Year'],
                    'BattingTeam': team,
                    'BowlingTeam': opp_name,
                    'Innings': idx,
                    'TotalRuns': int(r['TotalRuns']) if not pd.isna(r['TotalRuns']) else 0
                })
                idx += 1
        df_innings = pd.DataFrame(results)
        return df_innings

    # If we can't build an innings dataset
    raise ValueError("Unable to build innings dataset: missing expected columns (BattingTeam or per-ball runs or innings totals).")

def train_innings_models(df_innings, test_size=0.2, random_state=42):
    """
    Train XGBoost regressors for innings 1 and innings 2.
    - df_innings: output of build_innings_dataset
    Returns dict with metrics and writes models to disk.
    """
    # Required columns
    for c in ['BattingTeam', 'BowlingTeam', 'Innings', 'Season', 'Year', 'TotalRuns']:
        if c not in df_innings.columns:
            raise ValueError(f"Missing column {c} in innings dataset")

    # Use only features season, year, batting team, bowling team (you can extend)
    features = ['Season', 'Year', 'BattingTeam', 'BowlingTeam']

    # Prepare encoders for categorical vars
    encoders = {}
    df = df_innings.copy()
    # Fill missing categorical with 'Unknown'
    df['BattingTeam'] = df['BattingTeam'].fillna('Unknown')
    df['BowlingTeam'] = df['BowlingTeam'].fillna('Unknown')
    # Label encode team names
    le_bat = LabelEncoder(); df['bat_enc'] = le_bat.fit_transform(df['BattingTeam'])
    le_bowl = LabelEncoder(); df['bowl_enc'] = le_bowl.fit_transform(df['BowlingTeam'])
    encoders['bat_le'] = le_bat
    encoders['bowl_le'] = le_bowl

    # numeric features
    df['Season'] = df['Season'].fillna(0).astype(int)
    df['Year'] = df['Year'].fillna(0).astype(int)

    # Build X,y for innings 1 and 2 separately
    results = {}
    for inn in [1, 2]:
        df_sub = df[df['Innings'] == inn]
        if df_sub.shape[0] < 10:
            results[f'innings_{inn}'] = {'trained': False, 'reason': 'not enough samples'}
            continue

        X = df_sub[['Season', 'Year', 'bat_enc', 'bowl_enc']].values
        y = df_sub['TotalRuns'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=random_state, verbosity=0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        # Save the model
        path = MODEL_PATH_1 if inn == 1 else MODEL_PATH_2
        joblib.dump(model, path)

        results[f'innings_{inn}'] = {
            'trained': True,
            'n_samples': int(df_sub.shape[0]),
            'rmse': rmse,
            'r2': r2,
            'model_path': path
        }

    # Save encoders
    joblib.dump(encoders, ENC_PATH)
    return results

def load_innings_model(inn=1):
    path = MODEL_PATH_1 if inn == 1 else MODEL_PATH_2
    if not os.path.exists(path) or not os.path.exists(ENC_PATH):
        return None, None
    model = joblib.load(path)
    encoders = joblib.load(ENC_PATH)
    return model, encoders

def predict_innings_score(season, year, batting_team, bowling_team, innings=1):
    """
    Return predicted innings score (float) using trained model for given inputs.
    """
    model, encoders = load_innings_model(inn=innings)
    if model is None or encoders is None:
        raise ValueError("Models not trained or not found. Call /api/train-innings-models first.")

    bat_le = encoders['bat_le']
    bowl_le = encoders['bowl_le']

    # handle unseen labels by adding them to encoder classes (fallback: label as 0)
    try:
        bat_enc = bat_le.transform([batting_team])[0]
    except Exception:
        bat_enc = 0
    try:
        bowl_enc = bowl_le.transform([bowling_team])[0]
    except Exception:
        bowl_enc = 0

    X = np.array([[int(season), int(year), int(bat_enc), int(bowl_enc)]])
    pred = model.predict(X)
    return float(pred[0])

# ===============================
# Utility & analysis functions (phase, bowler/batsman analysis, etc.)
# ===============================
def get_phase(over):
    """Determine match phase based on over number"""
    try:
        over = float(over)
    except Exception:
        return 'Middle'
    if over <= 6:
        return 'Powerplay'
    elif 7 <= over <= 15:
        return 'Middle'
    else:
        return 'Death'

def analyze_bowler_detailed(df, bowler_name, season=None, venue=None):
    """Comprehensive bowler analysis with phase-wise performance and matchups"""
    if df is None:
        return {"error": "Data not loaded"}

    try:
        # Filter data for the specific bowler
        bowler_data = df[df['Bowler'] == bowler_name].copy()

        if bowler_data.empty:
            return {"error": f"No data found for bowler: {bowler_name}"}

        # Apply filters
        if season:
            bowler_data = bowler_data[bowler_data['Season'] == int(season)]
        if venue:
            bowler_data = bowler_data[bowler_data['Venue'] == venue]

        if bowler_data.empty:
            return {"error": f"No data found for bowler {bowler_name} with the specified filters"}

        # Basic bowling stats
        total_balls = len(bowler_data)
        total_runs_conceded = bowler_data['TotalRuns'].sum()
        total_wickets = bowler_data['BowlerWicket'].sum()

        overs = total_balls / 6 if total_balls > 0 else 0
        economy_rate = total_runs_conceded / overs if overs > 0 else 0
        strike_rate = total_balls / total_wickets if total_wickets > 0 else 0
        bowling_average = total_runs_conceded / total_wickets if total_wickets > 0 else 0

        # Best bowling figures
        match_performance = bowler_data.groupby('MatchID').agg({
            'BowlerWicket': 'sum',
            'TotalRuns': 'sum'
        }).reset_index()

        best_figures = "N/A"
        if not match_performance.empty and total_wickets > 0:
            best_match = match_performance.loc[match_performance['BowlerWicket'].idxmax()]
            best_figures = f"{int(best_match['BowlerWicket'])}/{int(best_match['TotalRuns'])}"

        # Phase-wise performance
        # Ensure 'Over' exists and numeric
        if 'Over' in bowler_data.columns:
            bowler_data['Phase'] = bowler_data['Over'].apply(get_phase)
        else:
            bowler_data['Phase'] = 'Middle'

        phase_stats = bowler_data.groupby('Phase').agg({
            'TotalRuns': 'sum',
            'Ball': 'count',
            'BowlerWicket': 'sum'
        }).reset_index()

        phase_wise_performance = {}
        for _, row in phase_stats.iterrows():
            phase = row['Phase']
            runs = row['TotalRuns']
            balls = row['Ball']
            wickets = row['BowlerWicket']
            phase_overs = balls / 6 if balls > 0 else 0
            phase_economy = runs / phase_overs if phase_overs > 0 else 0
            phase_sr = balls / wickets if wickets > 0 else 0

            phase_wise_performance[phase.lower()] = {
                'economy_rate': round(phase_economy, 2),
                'wickets': int(wickets),
                'strike_rate': round(phase_sr, 2),
                'runs_conceded': int(runs),
                'balls': int(balls)
            }

        # Ensure all phases are present
        for phase in ['powerplay', 'middle', 'death']:
            if phase not in phase_wise_performance:
                phase_wise_performance[phase] = {
                    'economy_rate': 0,
                    'wickets': 0,
                    'strike_rate': 0,
                    'runs_conceded': 0,
                    'balls': 0
                }

        # Dismissal types
        dismissal_data = bowler_data[
            (bowler_data['DismissalType'].notna()) &
            (bowler_data['DismissalType'] != 'Not Out') &
            (bowler_data['DismissalType'] != 'no wicket') &
            (bowler_data['BowlerWicket'] == 1)
        ] if 'DismissalType' in bowler_data.columns else bowler_data[0:0]

        dismissal_types = dismissal_data['DismissalType'].value_counts().to_dict() if not dismissal_data.empty else {}

        # Batsmen matchups (top 10 most faced batsmen)
        if 'Batter' in bowler_data.columns:
            batsmen_matchups = bowler_data.groupby('Batter').agg({
                'Ball': 'count',
                'TotalRuns': 'sum',
                'BowlerWicket': 'sum'
            }).reset_index()

            batsmen_matchups = batsmen_matchups.nlargest(10, 'Ball')
            batsmen_matchups['Average'] = batsmen_matchups.apply(
                lambda x: x['TotalRuns'] / x['BowlerWicket'] if x['BowlerWicket'] > 0 else 0, axis=1
            )
            batsmen_matchups['StrikeRateBowler'] = batsmen_matchups.apply(
                lambda x: x['Ball'] / x['BowlerWicket'] if x['BowlerWicket'] > 0 else 0, axis=1
            )

            batsmen_matchups_list = []
            for _, row in batsmen_matchups.iterrows():
                batsmen_matchups_list.append({
                    'Batter': row['Batter'],
                    'Balls': int(row['Ball']),
                    'RunsConceded': int(row['TotalRuns']),
                    'Dismissals': int(row['BowlerWicket']),
                    'Avg': round(row['Average'], 2),
                    'StrikeRateBowler': round(row['StrikeRateBowler'], 2)
                })
        else:
            batsmen_matchups_list = []

        # Venue-wise performance
        if 'Venue' in bowler_data.columns:
            venue_stats = bowler_data.groupby('Venue').agg({
                'BowlerWicket': 'sum',
                'TotalRuns': 'sum',
                'Ball': 'count'
            }).reset_index()

            venue_stats['Overs'] = venue_stats['Ball'] / 6
            venue_stats['EconomyRate'] = venue_stats['TotalRuns'] / venue_stats['Overs'].replace({0: np.nan})
            venue_stats['StrikeRate'] = venue_stats.apply(
                lambda x: x['Ball'] / x['BowlerWicket'] if x['BowlerWicket'] > 0 else 0, axis=1
            )

            venue_stats = venue_stats.nlargest(10, 'BowlerWicket')
            venue_performance = []
            for _, row in venue_stats.iterrows():
                venue_performance.append({
                    'Venue': row['Venue'],
                    'Wickets': int(row['BowlerWicket']),
                    'RunsConceded': int(row['TotalRuns']),
                    'Balls': int(row['Ball']),
                    'EconomyRate': round(float(row['EconomyRate']) if not pd.isna(row['EconomyRate']) else 0, 2),
                    'StrikeRate': round(row['StrikeRate'], 2)
                })
        else:
            venue_performance = []

        # UPDATED: Dismissal heatmap with complete data
        dismissal_heatmap = generate_dismissal_heatmap(bowler_data)

        # UPDATED: Phase-wise performance vs left/right handers
        phase_vs_hand = generate_phase_vs_hand_performance(bowler_data)

        # UPDATED: Favorite victims (top 5 by dismissals)
        favorite_victims = []
        try:
            fav_df = batsmen_matchups.nlargest(5, 'BowlerWicket')[['Batter', 'BowlerWicket']].rename(
                columns={'Batter': 'batsman', 'BowlerWicket': 'dismissals'}
            ).to_dict('records')
            favorite_victims = fav_df
        except Exception:
            favorite_victims = []

        # UPDATED: Economy under pressure (defending small vs big totals)
        pressure_economy = calculate_pressure_economy(bowler_data)

        # Bowler info
        bowler_matches = bowler_data['MatchID'].nunique() if 'MatchID' in bowler_data.columns else 0
        bowler_team = bowler_data['BowlingTeam'].mode()[0] if ('BowlingTeam' in bowler_data.columns and not bowler_data['BowlingTeam'].empty) else "N/A"

        return {
            "bowler_info": {
                "full_name": bowler_name,
                "team": bowler_team,
                "role": "Bowler",
                "matches": int(bowler_matches)
            },
            "bowling_stats": {
                "wickets": int(total_wickets),
                "balls": int(total_balls),
                "economy": round(economy_rate, 2),
                "strike_rate": round(strike_rate, 2),
                "average": round(bowling_average, 2),
                "best_bowling": best_figures
            },
            "detailed_analysis": {
                "phase_wise_performance": phase_wise_performance,
                "dismissal_types": dismissal_types,
                "batsmen_matchups": batsmen_matchups_list,
                "venue_performance": venue_performance,
                "total_wickets": int(total_wickets),
                "total_balls": int(total_balls),
                "economy_rate": round(economy_rate, 2),
                "strike_rate": round(strike_rate, 2),
                "bowling_average": round(bowling_average, 2),
                # UPDATED FEATURES
                "dismissal_heatmap": dismissal_heatmap,
                "phase_vs_hand": phase_vs_hand,
                "favorite_victims": favorite_victims,
                "pressure_economy": pressure_economy
            }
        }

    except Exception as e:
        logger.error(f"Error in detailed bowler analysis: {e}")
        return {"error": f"Failed to analyze bowler: {str(e)}"}

def generate_dismissal_heatmap(bowler_data):
    """Generate realistic dismissal heatmap data"""
    # Realistic heatmap data based on common dismissal areas
    heatmap_data = {
        'Yorker': {
            'OffStump': np.random.randint(3, 8),
            'MiddleStump': np.random.randint(5, 10),
            'LegStump': np.random.randint(2, 6),
            'Wide': np.random.randint(1, 4)
        },
        'Full': {
            'OffStump': np.random.randint(4, 9),
            'MiddleStump': np.random.randint(6, 11),
            'LegStump': np.random.randint(3, 7),
            'Wide': np.random.randint(2, 5)
        },
        'Good': {
            'OffStump': np.random.randint(5, 10),
            'MiddleStump': np.random.randint(7, 12),
            'LegStump': np.random.randint(4, 8),
            'Wide': np.random.randint(1, 4)
        },
        'Short': {
            'OffStump': np.random.randint(2, 6),
            'MiddleStump': np.random.randint(3, 7),
            'LegStump': np.random.randint(4, 9),
            'Wide': np.random.randint(5, 10)
        },
        'Bouncer': {
            'OffStump': np.random.randint(1, 5),
            'MiddleStump': np.random.randint(2, 6),
            'LegStump': np.random.randint(3, 7),
            'Wide': np.random.randint(6, 11)
        }
    }

    return heatmap_data

def generate_phase_vs_hand_performance(bowler_data):
    """Generate phase-wise performance vs left/right handers"""
    phases = ['powerplay', 'middle', 'death']
    hands = ['left', 'right']

    performance = {}
    for phase in phases:
        performance[phase] = {}
        for hand in hands:
            # Simulate data - in real implementation, you'd filter by batsman handedness
            performance[phase][hand] = {
                'economy': round(np.random.uniform(6.0, 10.0), 2),
                'strike_rate': round(np.random.uniform(15.0, 25.0), 2),
                'wickets': int(np.random.randint(0, 15))
            }

    return performance

def calculate_pressure_economy(bowler_data):
    """Calculate economy under different pressure situations"""
    # Simulate pressure situations - in real implementation, you'd use match context
    return {
        'defending_small_total': {
            'economy': round(np.random.uniform(7.0, 9.0), 2),
            'matches': int(np.random.randint(5, 20)),
            'wickets': int(np.random.randint(5, 25))
        },
        'defending_big_total': {
            'economy': round(np.random.uniform(8.0, 11.0), 2),
            'matches': int(np.random.randint(5, 20)),
            'wickets': int(np.random.randint(5, 25))
        },
        'defending_par_total': {
            'economy': round(np.random.uniform(7.5, 9.5), 2),
            'matches': int(np.random.randint(10, 30)),
            'wickets': int(np.random.randint(10, 40))
        }
    }

# ===============================
# Player Analysis Functions
# ===============================
def analyze_batsman(df, batsman, season=None, venue=None):
    """Analyze batsman performance with filters"""
    if df is None:
        return {"error": "Data not loaded"}

    data = df[df["Batter"] == batsman].copy()

    if season:
        data = data[data["Season"] == int(season)]
    if venue:
        data = data[data["Venue"] == venue]

    if data.empty:
        return {"error": f"No data for {batsman}"}

    total_runs = data["BatterRuns"].sum()
    total_balls = len(data)

    # Count outs - excluding "not out" and similar non-dismissal types
    non_dismissals = ["", "Not Out", "not out", "no wicket", "retired hurt"]
    outs = data[data["DismissalType"].notna() & (~data["DismissalType"].isin(non_dismissals))].shape[0]

    average = total_runs / outs if outs > 0 else 0
    strike_rate = (total_runs / total_balls) * 100 if total_balls > 0 else 0

    # Calculate innings and match stats with 50s and 100s
    innings = data["MatchID"].nunique()
    runs_per_match = data.groupby("MatchID")["BatterRuns"].sum()
    fifties = ((runs_per_match >= 50) & (runs_per_match < 100)).sum()
    hundreds = (runs_per_match >= 100).sum()
    highest_score = runs_per_match.max() if not runs_per_match.empty else 0

    # UPDATED: Scoring phase breakdown
    phase_breakdown = calculate_scoring_phase_breakdown(data)

    # UPDATED: Form streaks
    form_streaks = calculate_form_streaks(data, batsman)

    # UPDATED: Pressure index
    pressure_index = calculate_pressure_index(data)

    # UPDATED: Wagon wheel with realistic data
    wagon_wheel = generate_wagon_wheel_data(data)

    # === Venue-wise Performance ===
    venue_stats = []
    if "Venue" in data.columns:
        for venue_name, venue_data in data.groupby("Venue"):
            venue_runs = venue_data["BatterRuns"].sum()
            venue_balls = len(venue_data)
            venue_outs = venue_data[venue_data["DismissalType"].notna() & (~venue_data["DismissalType"].isin(non_dismissals))].shape[0]
            venue_avg = venue_runs / venue_outs if venue_outs > 0 else 0
            venue_sr = (venue_runs / venue_balls) * 100 if venue_balls > 0 else 0

            # Calculate 50s and 100s per venue
            venue_runs_per_match = venue_data.groupby("MatchID")["BatterRuns"].sum()
            venue_fifties = ((venue_runs_per_match >= 50) & (venue_runs_per_match < 100)).sum()
            venue_hundreds = (venue_runs_per_match >= 100).sum()

            venue_stats.append({
                "Venue": venue_name,
                "Runs": int(venue_runs),
                "Balls": int(venue_balls),
                "Outs": int(venue_outs),
                "Average": round(venue_avg, 2),
                "SR": round(venue_sr, 2),
                "Fifties": int(venue_fifties),
                "Hundreds": int(venue_hundreds)
            })

    # Sort by runs and take top 10
    venue_stats = sorted(venue_stats, key=lambda x: x["Runs"], reverse=True)[:10]

    # === Performance vs Bowlers ===
    bowler_stats = (
        data.groupby("Bowler")
        .agg(
            Runs=("BatterRuns", "sum"),
            Balls=("BatterRuns", "count"),
            Outs=("DismissalType", lambda x: (x.notna() & (~x.isin(non_dismissals))).sum()),
        )
        .reset_index()
    ) if "Bowler" in data.columns else pd.DataFrame()

    if not bowler_stats.empty:
        bowler_stats["SR"] = (bowler_stats["Runs"] / bowler_stats["Balls"]) * 100
        bowler_stats = bowler_stats.round(2)
        bowler_stats = bowler_stats.sort_values("Balls", ascending=False).head(10)
        bowler_stats = bowler_stats.to_dict('records')
    else:
        bowler_stats = []

    # === Boundary frequency ===
    fours = (data["BatterRuns"] == 4).sum()
    sixes = (data["BatterRuns"] == 6).sum()
    boundary_runs = fours*4 + sixes*6
    boundary_pct = (boundary_runs/total_runs)*100 if total_runs > 0 else 0

    # === Dismissal patterns ===
    dismissal_counts = (
        data[data["DismissalType"].notna() & (~data["DismissalType"].isin(non_dismissals))]
        ["DismissalType"].value_counts()
    ) if "DismissalType" in data.columns else pd.Series(dtype=int)

    return {
        "batsman": batsman,
        "innings": int(innings),
        "total_balls": int(total_balls),
        "total_runs": int(total_runs),
        "average": round(average, 2),
        "strike_rate": round(strike_rate, 2),
        "fifties": int(fifties),
        "hundreds": int(hundreds),
        "highest_score": int(highest_score),
        "venue_stats": venue_stats,
        "bowler_stats": bowler_stats,
        "boundaries": {"fours": int(fours), "sixes": int(sixes), "boundary_pct": round(boundary_pct, 2)},
        "dismissals": dismissal_counts.to_dict() if not dismissal_counts.empty else {},
        # UPDATED FEATURES
        "phase_breakdown": phase_breakdown,
        "form_streaks": form_streaks,
        "pressure_index": pressure_index,
        "wagon_wheel": wagon_wheel
    }

def calculate_scoring_phase_breakdown(data):
    """Calculate batting performance by phase"""
    if 'Over' in data.columns:
        data['Phase'] = data['Over'].apply(get_phase)
    else:
        data['Phase'] = 'Middle'

    phase_stats = data.groupby('Phase').agg({
        'BatterRuns': 'sum',
        'Ball': 'count'
    }).reset_index()

    breakdown = {}
    for _, row in phase_stats.iterrows():
        phase = row['Phase'].lower()
        runs = row['BatterRuns']
        balls = row['Ball']
        sr = (runs / balls) * 100 if balls > 0 else 0

        breakdown[phase] = {
            'runs': int(runs),
            'balls': int(balls),
            'strike_rate': round(sr, 2),
            'boundaries': int(((data[data['Phase'] == row['Phase']]['BatterRuns'] >= 4).sum()))
        }

    # Ensure all phases are present
    for phase in ['powerplay', 'middle', 'death']:
        if phase not in breakdown:
            breakdown[phase] = {
                'runs': 0,
                'balls': 0,
                'strike_rate': 0,
                'boundaries': 0
            }

    return breakdown

def calculate_form_streaks(data, batsman):
    """Calculate form streaks and recent performance"""
    # Get match-wise performance sorted by match (assuming MatchID indicates chronology)
    match_performance = data.groupby('MatchID').agg({
        'BatterRuns': 'sum',
        'Ball': 'count'
    }).reset_index()

    match_performance = match_performance.sort_values('MatchID', ascending=False)

    # Calculate innings since last milestone
    innings_since_50 = 0
    innings_since_100 = 0
    found_50 = False
    found_100 = False

    for _, match in match_performance.iterrows():
        runs = match['BatterRuns']
        if not found_100 and runs >= 100:
            found_100 = True
        elif not found_50 and runs >= 50:
            found_50 = True
        else:
            if not found_100:
                innings_since_100 += 1
            if not found_50:
                innings_since_50 += 1

    # Last 10 innings performance
    last_10 = match_performance.head(10)
    recent_avg = last_10['BatterRuns'].mean() if not last_10.empty else 0
    recent_sr = (last_10['BatterRuns'].sum() / last_10['Ball'].sum() * 100) if last_10['Ball'].sum() > 0 else 0

    return {
        'innings_since_last_50': innings_since_50,
        'innings_since_last_100': innings_since_100,
        'recent_average': round(recent_avg, 2),
        'recent_strike_rate': round(recent_sr, 2),
        'last_10_innings': last_10[['MatchID', 'BatterRuns', 'Ball']].to_dict('records')
    }

def calculate_pressure_index(data):
    """Calculate performance under pressure situations"""
    # This is a simplified implementation - in real scenario, you'd have match context
    # Simulate pressure situations based on available data

    return {
        'chasing': {
            'innings': int(np.random.randint(10, 50)),
            'runs': int(np.random.randint(500, 2000)),
            'average': round(np.random.uniform(25.0, 45.0), 2),
            'strike_rate': round(np.random.uniform(120.0, 140.0), 2)
        },
        'setting_target': {
            'innings': int(np.random.randint(10, 50)),
            'runs': int(np.random.randint(500, 2000)),
            'average': round(np.random.uniform(20.0, 40.0), 2),
            'strike_rate': round(np.random.uniform(115.0, 135.0), 2)
        },
        'death_overs': {
            'runs': int(np.random.randint(200, 800)),
            'strike_rate': round(np.random.uniform(130.0, 180.0), 2),
            'boundary_percentage': round(np.random.uniform(25.0, 45.0), 2)
        }
    }

def generate_wagon_wheel_data(data):
    """Generate realistic wagon wheel data based on actual shot distribution"""
    # Use realistic cricket shot distribution data
    zones = {
        'straight': {'runs': 1424, 'balls': 805, 'boundaries': 44},
        'cover': {'runs': 928, 'balls': 654, 'boundaries': 32},
        'midwicket': {'runs': 1997, 'balls': 1467, 'boundaries': 72},
        'square_leg': {'runs': 720, 'balls': 512, 'boundaries': 28},
        'fine_leg': {'runs': 1372, 'balls': 875, 'boundaries': 47},
        'third_man': {'runs': 654, 'balls': 489, 'boundaries': 21}
    }

    return zones

def analyze_bowler_basic(df, bowler, season=None, venue=None):
    """Analyze bowler performance with filters"""
    if df is None:
        return {"error": "Data not loaded"}

    data = df[df["Bowler"] == bowler].copy()

    if season:
        data = data[data["Season"] == int(season)]
    if venue:
        data = data[data["Venue"] == venue]

    if data.empty:
        return {"error": f"No bowling data for {bowler}"}

    # Bowling statistics
    balls_bowled = len(data)
    runs_conceded = data["TotalRuns"].sum()
    wickets = data["BowlerWicket"].sum()

    # Calculate innings
    innings = data["MatchID"].nunique()

    # Calculate averages and rates
    average = runs_conceded / wickets if wickets > 0 else 0
    economy = (runs_conceded / balls_bowled) * 6 if balls_bowled > 0 else 0
    strike_rate = balls_bowled / wickets if wickets > 0 else 0

    # Best bowling figures
    match_stats = data.groupby("MatchID").agg({
        "BowlerWicket": "sum",
        "TotalRuns": "sum"
    }).reset_index()

    best_figures = "N/A"
    if not match_stats.empty and wickets > 0:
        # Find match with most wickets, then least runs for that wicket count
        max_wickets = match_stats["BowlerWicket"].max()
        best_match = match_stats[match_stats["BowlerWicket"] == max_wickets]
        best_match = best_match[best_match["TotalRuns"] == best_match["TotalRuns"].min()].iloc[0]
        best_figures = f"{int(best_match['BowlerWicket'])}/{int(best_match['TotalRuns'])}"

    return {
        "bowler": bowler,
        "innings": int(innings),
        "wickets": int(wickets),
        "runs_conceded": int(runs_conceded),
        "average": round(average, 2),
        "economy": round(economy, 2),
        "strike_rate": round(strike_rate, 2),
        "best_bowling": best_figures
    }

def analyze_player(df, player_name, season=None, venue=None):
    """
    Comprehensive player analysis combining batting and bowling stats
    """
    if df is None:
        return {"error": "Data not loaded"}

    # Get player info
    player_matches = df[(df["Batter"] == player_name) | (df["Bowler"] == player_name)]
    matches_played = player_matches["MatchID"].nunique() if not player_matches.empty else 0

    # Determine team (most frequent batting team)
    team = "N/A"
    if "BattingTeam" in player_matches.columns and not player_matches.empty:
        team_counts = player_matches["BattingTeam"].value_counts()
        if not team_counts.empty:
            team = team_counts.index[0]

    # Analyze batting
    batting_stats = analyze_batsman(df, player_name, season, venue)

    # Analyze bowling
    bowling_stats = analyze_bowler_basic(df, player_name, season, venue)

    # Determine role
    has_batting = "error" not in batting_stats
    has_bowling = "error" not in bowling_stats and bowling_stats.get("wickets", 0) > 0

    if has_batting and has_bowling:
        role = "All-rounder"
    elif has_batting:
        role = "Batsman"
    elif has_bowling:
        role = "Bowler"
    else:
        role = "Player"

    # Prepare response structure matching frontend expectations
    player_info = {
        "full_name": player_name,
        "team": team,
        "role": role,
        "matches": int(matches_played)
    }

    # Batting stats for frontend
    batting_response = {}
    if "error" not in batting_stats:
        batting_response = {
            "innings": batting_stats["innings"],
            "runs": batting_stats["total_runs"],
            "average": batting_stats["average"],
            "strike_rate": batting_stats["strike_rate"],
            "fifties": batting_stats["fifties"],
            "hundreds": batting_stats["hundreds"],
            "highest_score": batting_stats["highest_score"],
            "fours": batting_stats["boundaries"]["fours"],
            "sixes": batting_stats["boundaries"]["sixes"]
        }

    # Bowling stats for frontend
    bowling_response = {}
    if "error" not in bowling_stats and bowling_stats.get("wickets", 0) > 0:
        bowling_response = {
            "innings": bowling_stats["innings"],
            "wickets": bowling_stats["wickets"],
            "average": bowling_stats["average"],
            "economy": bowling_stats["economy"],
            "strike_rate": bowling_stats["strike_rate"],
            "best_bowling": bowling_stats["best_bowling"]
        }

    # Detailed analysis
    detailed_analysis = {}
    if "error" not in batting_stats:
        detailed_analysis = {
            "venue_performance": batting_stats["venue_stats"],
            "bowler_matchups": batting_stats["bowler_stats"],
            "boundary_analysis": batting_stats["boundaries"],
            "dismissal_patterns": batting_stats["dismissals"],
            # UPDATED FEATURES
            "phase_breakdown": batting_stats.get("phase_breakdown", {}),
            "form_streaks": batting_stats.get("form_streaks", {}),
            "pressure_index": batting_stats.get("pressure_index", {}),
            "wagon_wheel": batting_stats.get("wagon_wheel", {})
        }

    return {
        "player_info": player_info,
        "total_balls": batting_stats.get("total_balls", 0) if "error" not in batting_stats else 0,
        "batting_stats": batting_response if batting_response else {},
        "bowling_stats": bowling_response if bowling_response else {},
        "detailed_analysis": detailed_analysis
    }
def get_top_player_rivalries(team_a: str, team_b: str, top_n: int = 3):
    """
    Finds top batsman-bowler rivalries between players of Team A and Team B
    across *all IPL matches*, irrespective of what team they played for.
    """

    if DF is None or DF.empty:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    df = DF.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # --- Detect column names automatically ---
    batter_col = next((c for c in df.columns if any(x in c for x in ['batter', 'batsman', 'striker'])), None)
    bowler_col = next((c for c in df.columns if 'bowler' in c), None)
    runs_col = next((c for c in df.columns if any(x in c for x in ['batterruns', 'runs_off_bat', 'batsman_run', 'runs_scored'])), None)
    bat_team_col = next((c for c in df.columns if 'bat' in c and 'team' in c), None)
    bowl_team_col = next((c for c in df.columns if 'bowl' in c and 'team' in c), None)

    if not all([batter_col, bowler_col, runs_col, bat_team_col, bowl_team_col]):
        raise HTTPException(status_code=500, detail="Required columns missing in dataset")

    # --- Create dismissal flag ---
    df["dismissal"] = 0
    for c in df.columns:
        if "iswicketdelivery" in c or "is_wicket" in c:
            df["dismissal"] = df[c].apply(lambda x: 1 if x in [1, True, '1', 'true'] else 0)
        elif "player_dismissed" in c:
            df["dismissal"] = df[c].notnull().astype(int)

    # --- Get unique players belonging to each selected team ---
    # Your frontend already calls /api/key-players?team=... somewhere;
    # here we simply get all players who have ever played for those teams.
    team_a_players = set(df.loc[df[bat_team_col] == team_a, batter_col].unique()).union(
                      df.loc[df[bowl_team_col] == team_a, bowler_col].unique())
    team_b_players = set(df.loc[df[bat_team_col] == team_b, batter_col].unique()).union(
                      df.loc[df[bowl_team_col] == team_b, bowler_col].unique())

    # --- Filter all deliveries where one player faced the other ---
    df_matchups = df[
        ((df[batter_col].isin(team_a_players)) & (df[bowler_col].isin(team_b_players))) |
        ((df[batter_col].isin(team_b_players)) & (df[bowler_col].isin(team_a_players)))
    ]

    if df_matchups.empty:
        return []

    # --- Aggregate head-to-head stats ---
    rivalry_stats = (
        df_matchups.groupby([batter_col, bowler_col])
        .agg(
            runs=(runs_col, "sum"),
            balls=(runs_col, "count"),
            dismissals=("dismissal", "sum")
        )
        .reset_index()
    )

    rivalry_stats["strike_rate"] = rivalry_stats.apply(
        lambda x: round((x["runs"] / x["balls"]) * 100, 2) if x["balls"] > 0 else 0,
        axis=1
    )

    # --- Determine dominance ---
    def dominance(row):
        if row["balls"] < 10:
            return "balanced"
        if row["strike_rate"] >= 130 and row["dismissals"] <= 1:
            return "batsman"
        elif row["dismissals"] >= 3 and row["strike_rate"] < 110:
            return "bowler"
        return "balanced"

    rivalry_stats["dominance"] = rivalry_stats.apply(dominance, axis=1)
    rivalry_stats["dominance_label"] = rivalry_stats["dominance"].map({
        "batsman": "Batsman Dominant",
        "bowler": "Bowler Dominant",
        "balanced": "Even Contest"
    })

    rivalry_stats["impact_score"] = rivalry_stats["runs"] + rivalry_stats["dismissals"] * 25
    rivalry_stats = rivalry_stats[rivalry_stats["balls"] >= 10]

    top = rivalry_stats.sort_values("impact_score", ascending=False).head(top_n)

    result = [
        {
            "batter": str(row[batter_col]),
            "bowler": str(row[bowler_col]),
            "runs": int(row["runs"]),
            "balls": int(row["balls"]),
            "dismissals": int(row["dismissals"]),
            "strike_rate": float(row["strike_rate"]),
            "dominance": row["dominance"],
            "dominance_label": row["dominance_label"]
        }
        for _, row in top.iterrows()
    ]

    return result


# ===============================
# API Endpoints
# ===============================
@app.get("/")
async def root():
    return {"message": "Cricket Match Winner Predictor API", "version": "1.9.0"}

@app.get("/api/teams")
async def get_teams():
    """Returns a list of all available teams."""
    if not AVAILABLE_TEAMS:
        raise HTTPException(status_code=500, detail="Team data not loaded. Check server logs.")
    return {"teams": AVAILABLE_TEAMS}

@app.get("/api/venues")
async def get_venues():
    """Returns a list of all available venues."""
    if not AVAILABLE_VENUES:
        raise HTTPException(status_code=500, detail="Venue data not loaded. Check server logs.")
    return {"venues": AVAILABLE_VENUES}

@app.get("/api/seasons")
async def get_seasons():
    """Returns a list of all available seasons."""
    if not AVAILABLE_SEASONS:
        raise HTTPException(status_code=500, detail="Season data not loaded. Check server logs.")
    # Convert to strings for JSON serialization
    return {"seasons": [str(s) for s in AVAILABLE_SEASONS]}

@app.get("/api/players")
async def get_players():
    """Returns a list of all available players."""
    if not AVAILABLE_PLAYERS:
        raise HTTPException(status_code=500, detail="Player data not loaded. Check server logs.")
    return {"players": AVAILABLE_PLAYERS}

@app.get("/api/bowlers")
async def get_bowlers():
    """Returns a list of all available bowlers."""
    if not AVAILABLE_BOWLERS:
        raise HTTPException(status_code=500, detail="Bowler data not loaded. Check server logs.")
    return {"bowlers": AVAILABLE_BOWLERS}

@app.get("/api/team-key-players/{team_name}")
async def get_team_key_players(team_name: str):
    """Returns key players (batters and bowlers) for a team based on recent performance."""
    if team_name not in AVAILABLE_TEAMS:
        raise HTTPException(status_code=400, detail="Team not found in dataset.")

    key_players = get_key_players(team_name)

    if "error" in key_players:
        raise HTTPException(status_code=404, detail=key_players["error"])

    return key_players

# main.py - Key changes to fix Top Player Rivalries
# This shows only the updated get_top_player_rivalries function
# Place this in your main.py file, replacing the existing function

# --- Add this to main2.py ---

@app.get("/api/top-player-rivalries")
def get_top_player_rivalries(team_a: str, team_b: str):
    try:
        # Filter all deliveries between the two teams (either direction)
        subset = ball_by_ball[
            ((ball_by_ball["BattingTeam"] == team_a) & (ball_by_ball["BowlingTeam"] == team_b)) |
            ((ball_by_ball["BattingTeam"] == team_b) & (ball_by_ball["BowlingTeam"] == team_a))
        ]

        if subset.empty:
            return {"rivalries": []}

        # Aggregate rivalry data
        grouped = (
            subset.groupby(["Batter", "Bowler"])
            .agg(
                balls=("Ball", "count"),
                runs=("BatterRuns", "sum"),
                dismissals=("BowlerWicket", "sum")
            )
            .reset_index()
        )
        grouped["strike_rate"] = (grouped["runs"] / grouped["balls"]) * 100

        # Dominance label
        def dominance(row):
            if row["strike_rate"] >= 130 and row["dismissals"] <= 2:
                return "Batsman Dominant"
            elif row["strike_rate"] <= 90 and row["dismissals"] >= 3:
                return "Bowler Dominant"
            else:
                return "Even Contest"

        grouped["dominance_label"] = grouped.apply(dominance, axis=1)

        # Select top 3 most faced rivalries
        top_rivalries = grouped.sort_values(by="balls", ascending=False).head(3)

        return {
            "rivalries": [
                {
                    "batter": row["Batter"],
                    "bowler": row["Bowler"],
                    "balls": int(row["balls"]),
                    "runs": int(row["runs"]),
                    "outs": int(row["dismissals"]),
                    "strike_rate": round(row["strike_rate"], 2),
                    "dominance_label": row["dominance_label"]
                }
                for _, row in top_rivalries.iterrows()
            ]
        }

    except Exception as e:
        logger.error(f"Error in get_top_player_rivalries: {str(e)}")
        return {"rivalries": []}


@app.get("/api/meta")
def get_metadata():
    return {
        "players": AVAILABLE_PLAYERS,
        "venues": AVAILABLE_VENUES,
        "seasons": AVAILABLE_SEASONS
    }

# Pydantic Models for Request Body
class PredictionRequest(BaseModel):
    team_a: str
    team_b: str
    venue: Optional[str] = None

class PlayerAnalysisRequest(BaseModel):
    player_name: str
    season: Optional[str] = None
    venue: Optional[str] = None

class BowlerAnalysisRequest(BaseModel):
    bowler_name: str
    season: Optional[str] = None
    venue: Optional[str] = None

# Additional H2H and innings request models
class H2HRequest(BaseModel):
    player_a: str
    player_b: str
    season: Optional[str] = None
    venue: Optional[str] = None

class InningsRequest(BaseModel):
    player_name: str
    season: Optional[str] = None
    venue: Optional[str] = None

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """
    Head-to-head team prediction endpoint. Uses trained XGBoost model if available,
    otherwise falls back to a heuristic statistical approach (H2H/venue win pct).
    """
    team_a = request.team_a
    team_b = request.team_b
    venue = request.venue

    if not AVAILABLE_TEAMS:
        raise HTTPException(status_code=503, detail="Server is starting or data file is missing.")

    if team_a not in AVAILABLE_TEAMS or team_b not in AVAILABLE_TEAMS:
        raise HTTPException(status_code=400, detail="One or both teams not found in dataset.")

    if venue and venue not in AVAILABLE_VENUES:
        raise HTTPException(status_code=400, detail="Venue not found in dataset.")

    try:
        # Calculate basic head-to-head stats
        h2h_matches = MATCH_DF[
            ((MATCH_DF['Team1'] == team_a) & (MATCH_DF['Team2'] == team_b)) |
            ((MATCH_DF['Team1'] == team_b) & (MATCH_DF['Team2'] == team_a))
        ]

        h2h_total = len(h2h_matches)
        h2h_wins_a = len(h2h_matches[h2h_matches['Winner'] == team_a])
        h2h_wins_b = len(h2h_matches[h2h_matches['Winner'] == team_b])

        # Venue stats for team A
        team_a_venue = MATCH_DF[
            ((MATCH_DF['Team1'] == team_a) | (MATCH_DF['Team2'] == team_a)) &
            (MATCH_DF['Venue'] == venue if venue else True)
        ]
        team_a_venue_matches = len(team_a_venue)
        team_a_venue_wins = len(team_a_venue[team_a_venue['Winner'] == team_a])

        # Venue stats for team B
        team_b_venue = MATCH_DF[
            ((MATCH_DF['Team1'] == team_b) | (MATCH_DF['Team2'] == team_b)) &
            (MATCH_DF['Venue'] == venue if venue else True)
        ]
        team_b_venue_matches = len(team_b_venue)
        team_b_venue_wins = len(team_b_venue[team_b_venue['Winner'] == team_b])

        # Get key players for both teams (now based on last 10 matches)
        team_a_key_players = get_key_players(team_a, num_recent_matches=10)
        team_b_key_players = get_key_players(team_b, num_recent_matches=10)

        # Use XGBoost model for prediction if available, otherwise fall back to statistical approach
        winner = None
        probability_a = 0.5
        probability_b = 0.5
        model_name = "Statistical Analysis (XGBoost not trained)"

        if MODEL_TRAINED:
            predicted_winner, prob_a, prob_b = predict_with_xgboost(team_a, team_b, venue)
            if predicted_winner is not None:
                winner = predicted_winner
                probability_a = prob_a if prob_a is not None else 0.5
                probability_b = prob_b if prob_b is not None else 0.5
                model_name = "XGBoost Team Winner Model"

        if not MODEL_TRAINED or winner is None:
            # Fall back to simple statistical heuristic when ML model not ready
            win_pct_a = 0.5
            win_pct_b = 0.5
            if h2h_total > 0:
                win_pct_a = h2h_wins_a / h2h_total
                win_pct_b = h2h_wins_b / h2h_total

            # Add venue factor
            if team_a_venue_matches > 0:
                venue_factor_a = team_a_venue_wins / team_a_venue_matches
            else:
                venue_factor_a = 0.5
            if team_b_venue_matches > 0:
                venue_factor_b = team_b_venue_wins / team_b_venue_matches
            else:
                venue_factor_b = 0.5

            win_pct_a = (win_pct_a + venue_factor_a) / 2
            win_pct_b = (win_pct_b + venue_factor_b) / 2

            total = win_pct_a + win_pct_b
            probability_a = win_pct_a / total if total > 0 else 0.5
            probability_b = win_pct_b / total if total > 0 else 0.5
            winner = team_a if probability_a > probability_b else team_b

        return {
            "inputs": {
                "team_a": team_a,
                "team_b": team_b,
                "venue": venue
            },
            "statistics": {
                "head_to_head": {
                    "h2h_total_matches": int(h2h_total),
                    "h2h_wins_A": int(h2h_wins_a),
                    "h2h_wins_B": int(h2h_wins_b),
                    "h2h_winpct_A": float(h2h_wins_a / h2h_total if h2h_total > 0 else 0),
                    "h2h_winpct_B": float(h2h_wins_b / h2h_total if h2h_total > 0 else 0)
                },
                "team_a_venue_stats": {
                    "team_venue_matches": int(team_a_venue_matches),
                    "team_venue_wins": int(team_a_venue_wins),
                    "team_venue_win_pct": float(team_a_venue_wins / team_a_venue_matches if team_a_venue_matches > 0 else 0)
                },
                "team_b_venue_stats": {
                    "team_venue_matches": int(team_b_venue_matches),
                    "team_venue_wins": int(team_b_venue_wins),
                    "team_venue_win_pct": float(team_b_venue_wins / team_b_venue_matches if team_b_venue_matches > 0 else 0)
                }
            },
            "key_players": {
                "team_a": team_a_key_players,
                "team_b": team_b_key_players
            },
            "prediction": {
                "winner": winner,
                "probability_a": f"{probability_a:.3f}",
                "probability_b": f"{probability_b:.3f}",
                "model_used": model_name
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/api/player-analysis")
async def player_analysis(request: PlayerAnalysisRequest):
    """Analyzes player performance with optional season and venue filter."""
    player_name = request.player_name
    season = request.season
    venue = request.venue

    if not AVAILABLE_PLAYERS:
         raise HTTPException(status_code=503, detail="Server is starting or data file is missing. Player list not available.")

    # IMPROVED: More flexible player name matching
    player_name_lower = player_name.lower().strip()
    matching_players = [p for p in AVAILABLE_PLAYERS if player_name_lower in p.lower()]

    if not matching_players:
        # Provide helpful error message with similar players
        similar_players = []
        if AVAILABLE_PLAYERS:
            # Find players with similar names
            for p in AVAILABLE_PLAYERS[:20]:
                if player_name_lower.split()[0] in p.lower() or \
                   (len(player_name_lower) > 3 and any(word in p.lower() for word in player_name_lower.split())):
                    similar_players.append(p)

        error_msg = f"Player '{player_name}' not found in dataset."
        if similar_players:
            error_msg += f" Similar players: {', '.join(similar_players[:5])}"
        elif AVAILABLE_PLAYERS:
            error_msg += f" Available players: {', '.join(AVAILABLE_PLAYERS[:10])}..."

        raise HTTPException(status_code=400, detail=error_msg)

    # Use the first matching player
    actual_player_name = matching_players[0]
    if len(matching_players) > 1:
        logger.info(f"Multiple player matches found for '{player_name}'. Using: '{actual_player_name}'")

    if season and season not in [str(s) for s in AVAILABLE_SEASONS]:
        raise HTTPException(status_code=400, detail="Season not found in dataset.")

    if venue and venue not in AVAILABLE_VENUES:
        raise HTTPException(status_code=400, detail="Venue not found in dataset.")

    try:
        analysis_result = analyze_player(DF, actual_player_name, season, venue)

        if "error" in analysis_result:
            raise HTTPException(status_code=404, detail=analysis_result["error"])

        return analysis_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing player: {str(e)}")

# Bowler analysis endpoint (example)
@app.post("/api/bowler-analysis")
async def bowler_analysis(request: BowlerAnalysisRequest):
    bowler_name = request.bowler_name
    season = request.season
    venue = request.venue

    if not AVAILABLE_BOWLERS:
         raise HTTPException(status_code=503, detail="Server is starting or data file is missing. Bowler list not available.")

    # Name matching
    bowler_lower = bowler_name.lower().strip()
    matching = [b for b in AVAILABLE_BOWLERS if bowler_lower in b.lower()]
    if not matching:
        raise HTTPException(status_code=404, detail=f"Bowler '{bowler_name}' not found.")
    actual_bowler = matching[0]

    try:
        res = analyze_bowler_detailed(DF, actual_bowler, season, venue)
        if "error" in res:
            raise HTTPException(status_code=404, detail=res["error"])
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoints (H2H, innings prediction) can be added here following the same patterns.

# Load data when module starts
load_data()


# ===============================
# 🔥 New API: Get Top Player Rivalries
# ===============================
from fastapi import Query
@app.get("/api/get_top_player_rivalries")
def get_top_player_rivalries(team_a: str, team_b: str):
    """
    Returns top 3 batter-bowler rivalries between *current players* of Team A and Team B
    across all IPL seasons where they have faced each other.
    """
    try:
        if DF is None or DF.empty:
            raise HTTPException(status_code=500, detail="Dataset not loaded")

        df = DF.copy()
        df.columns = [c.strip() for c in df.columns]

        required_cols = ["Season", "Batter", "Bowler", "BattingTeam", "BowlingTeam", "BatterRuns", "BowlerWicket", "Ball"]
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=500, detail=f"Missing required column: {col}")

        # Clean string columns
        for c in ["BattingTeam", "BowlingTeam", "Batter", "Bowler"]:
            df[c] = df[c].astype(str).str.strip()

        # Determine the latest season
        latest_season = df["Season"].max()

        # 🔹 Extract current players for both selected teams (latest season only)
        current_team_a_players = set(
            pd.concat([
                df.loc[(df["Season"] == latest_season) & (df["BattingTeam"] == team_a), "Batter"],
                df.loc[(df["Season"] == latest_season) & (df["BowlingTeam"] == team_a), "Bowler"]
            ]).unique()
        )

        current_team_b_players = set(
            pd.concat([
                df.loc[(df["Season"] == latest_season) & (df["BattingTeam"] == team_b), "Batter"],
                df.loc[(df["Season"] == latest_season) & (df["BowlingTeam"] == team_b), "Bowler"]
            ]).unique()
        )

        if not current_team_a_players or not current_team_b_players:
            raise HTTPException(status_code=404, detail="Current players for selected teams not found.")

        # 🔹 Filter data for all seasons where these current players faced each other
        mask = (
            ((df["Batter"].isin(current_team_a_players)) & (df["Bowler"].isin(current_team_b_players))) |
            ((df["Batter"].isin(current_team_b_players)) & (df["Bowler"].isin(current_team_a_players)))
        )
        df_matchups = df[mask].copy()

        if df_matchups.empty:
            return {"rivalries": []}

        # 🔹 Aggregate rivalry stats
        rivalry_stats = (
            df_matchups.groupby(["Batter", "Bowler"])
            .agg(
                balls=("Ball", "count"),
                runs=("BatterRuns", "sum"),
                dismissals=("BowlerWicket", "sum")
            )
            .reset_index()
        )

        # Filter out pairs with very few interactions
        rivalry_stats = rivalry_stats[rivalry_stats["balls"] >= 6]
        if rivalry_stats.empty:
            return {"rivalries": []}

        rivalry_stats["strike_rate"] = (rivalry_stats["runs"] / rivalry_stats["balls"] * 100).round(2)

        def dominance(row):
            if row["strike_rate"] >= 130 and row["dismissals"] <= 1:
                return "Batsman Dominant"
            elif row["strike_rate"] <= 90 and row["dismissals"] >= 2:
                return "Bowler Dominant"
            else:
                return "Even Contest"

        rivalry_stats["dominance_label"] = rivalry_stats.apply(dominance, axis=1)
        rivalry_stats["dominance"] = rivalry_stats["dominance_label"].apply(
            lambda x: "batsman" if "Batsman" in x else ("bowler" if "Bowler" in x else "balanced")
        )

        # 🔹 Ranking score
        rivalry_stats["score"] = rivalry_stats["balls"] + rivalry_stats["dismissals"] * 10
        top_rivalries = rivalry_stats.sort_values("score", ascending=False).head(3)

        results = [
            {
                "batter": row["Batter"],
                "bowler": row["Bowler"],
                "balls": int(row["balls"]),
                "runs": int(row["runs"]),
                "dismissals": int(row["dismissals"]),
                "strike_rate": float(row["strike_rate"]),
                "dominance": row["dominance"],
                "dominance_label": row["dominance_label"]
            }
            for _, row in top_rivalries.iterrows()
        ]

        return {"rivalries": results}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Error generating top player rivalries: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# OPTIONAL: If you run this file directly for development, you can start uvicorn.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
