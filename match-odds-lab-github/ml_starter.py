"""
Match Odds Lab - machine learning starter
=========================================

Run it from the project folder (the one with model.py in it):

    pip install pandas scikit-learn lightgbm
    python ml/ml_starter.py

What it does, in the same order as the step-by-step guide:

    Step 1  load the 20 years of league matches
    Step 2  build features: what was known about each match BEFORE kick-off
    Step 3  set the answers to learn (result, over 2.5 goals, both teams scored)
    Step 4  split by time: learn from old seasons, test on the last three
    Step 5  train a gradient-boosting model (LightGBM)
    Step 6  score it against the bookmakers, the plain averages and your current models

It doesn't change the website or the existing models. It only prints a report.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore", category=FutureWarning)        # library housekeeping messages
warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

HERE = Path(__file__).resolve().parent.parent      # the project folder
MATCHES = HERE / "data" / "matches.csv"
BACKTEST = HERE / "data" / "backtest.json"

TEST_SEASONS = ["2023/24", "2024/25", "2025/26"]   # the same seasons the current backtest uses
VALID_SEASON = "2022/23"                           # used to decide when to stop training
FIRST_TRAIN_SEASON = "2007/08"                     # 2006/07 only warms up the form numbers


# ---------------------------------------------------------------- Step 1: load
def load():
    if not MATCHES.exists():
        sys.exit(f"Can't find {MATCHES}. Run python download_data.py --current first.")
    df = pd.read_csv(MATCHES, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["FTHG", "FTAG"]).sort_values(["Date", "Div", "HomeTeam"]).reset_index(drop=True)
    df["match_id"] = np.arange(len(df))

    def first_available(*cols):
        """Average odds where the file has them, otherwise older column names."""
        out = pd.Series(np.nan, index=df.index)
        for c in cols:
            if c in df:
                out = out.fillna(pd.to_numeric(df[c], errors="coerce"))
        return out

    df["oH"] = first_available("AvgH", "BbAvH", "B365H")
    df["oD"] = first_available("AvgD", "BbAvD", "B365D")
    df["oA"] = first_available("AvgA", "BbAvA", "B365A")
    df["oO"] = first_available("Avg>2.5", "BbAv>2.5", "B365>2.5")
    df["oU"] = first_available("Avg<2.5", "BbAv<2.5", "B365<2.5")
    print(f"Step 1: {len(df):,} matches from {df.Date.min():%d %b %Y} to {df.Date.max():%d %b %Y}")
    return df


# ------------------------------------------------------- Step 2: features
def add_elo(df, k=20.0, hfa=60.0, regress=0.15, start=1500.0, promoted=1450.0):
    """Elo ratings as they stood BEFORE each match (so no peeking at the result)."""
    elo, season_of = {}, {}
    pre_h, pre_a = np.empty(len(df)), np.empty(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        for team in (r.HomeTeam, r.AwayTeam):
            if team not in elo:
                elo[team] = start if r.Season == df.Season.iloc[0] else promoted
            elif season_of[team] != r.Season:          # new season: pull back toward average
                elo[team] = start + (1 - regress) * (elo[team] - start)
            season_of[team] = r.Season
        h, a = elo[r.HomeTeam], elo[r.AwayTeam]
        pre_h[i], pre_a[i] = h, a
        expected = 1 / (1 + 10 ** ((a - h - hfa) / 400))
        actual = 1.0 if r.FTHG > r.FTAG else 0.5 if r.FTHG == r.FTAG else 0.0
        margin = np.log(abs(r.FTHG - r.FTAG) + 1) + 1    # bigger wins move ratings more
        elo[r.HomeTeam] = h + k * margin * (actual - expected)
        elo[r.AwayTeam] = a - k * margin * (actual - expected)
    df["elo_home"], df["elo_away"] = pre_h, pre_a
    df["elo_diff"] = pre_h - pre_a
    return df


def add_form(df):
    """Recent form for each team. shift(1) means only matches BEFORE this one count."""
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
    home = pd.DataFrame({"match_id": df.match_id, "team": df.HomeTeam, "date": df.Date, "season": df.Season,
                         "is_home": 1, "gf": df.FTHG, "ga": df.FTAG, "sf": num("HS"), "sa": num("AS"),
                         "tf": num("HST"), "ta": num("AST"), "cf": num("HC"), "ca": num("AC")})
    away = pd.DataFrame({"match_id": df.match_id, "team": df.AwayTeam, "date": df.Date, "season": df.Season,
                         "is_home": 0, "gf": df.FTAG, "ga": df.FTHG, "sf": num("AS"), "sa": num("HS"),
                         "tf": num("AST"), "ta": num("HST"), "cf": num("AC"), "ca": num("HC")})
    long = pd.concat([home, away]).sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    long["pts"] = np.where(long.gf > long.ga, 3, np.where(long.gf == long.ga, 1, 0))
    long["o25"] = ((long.gf + long.ga) > 2.5).astype(float)
    long["btts"] = ((long.gf > 0) & (long.ga > 0)).astype(float)

    g = long.groupby("team")
    feats = []
    for n in (5, 10):
        for col in ("gf", "ga", "sf", "sa", "tf", "ta", "cf", "ca", "pts", "o25", "btts"):
            name = f"{col}_{n}"
            long[name] = g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=3).mean())
            feats.append(name)
    long["rest_days"] = g["date"].diff().dt.days.clip(upper=30)
    long["played_this_season"] = long.groupby(["team", "season"]).cumcount()
    feats += ["rest_days", "played_this_season"]

    for side, flag in (("h", 1), ("a", 0)):
        part = long[long.is_home == flag].set_index("match_id")[feats].add_prefix(f"{side}_")
        df = df.join(part, on="match_id")
    for col in ("gf_10", "ga_10", "tf_10", "ta_10", "pts_10"):
        df[f"diff_{col}"] = df[f"h_{col}"] - df[f"a_{col}"]
    return df


def add_odds_features(df):
    """Bookmaker odds turned into probabilities, with their margin taken out."""
    inv = 1 / df[["oH", "oD", "oA"]]
    p = inv.div(inv.sum(axis=1), axis=0)
    df["book_H"], df["book_D"], df["book_A"] = p.oH, p.oD, p.oA
    inv2 = 1 / df[["oO", "oU"]]
    df["book_O25"] = (inv2.oO / inv2.sum(axis=1))
    return df


# ---------------------------------------------------------- Steps 3 to 6
def main():
    df = load()
    print("Step 2: building features (Elo, form over the last 5 and 10 matches, rest days, odds)...")
    df = add_elo(df)
    df = add_form(df)
    df = add_odds_features(df).copy()
    df["league"] = df.Div.astype("category")

    # Step 3: the answers to learn
    df["y_result"] = np.where(df.FTHG > df.FTAG, 0, np.where(df.FTHG == df.FTAG, 1, 2))   # 0 home, 1 draw, 2 away
    df["y_o25"] = ((df.FTHG + df.FTAG) > 2.5).astype(int)
    df["y_btts"] = ((df.FTHG > 0) & (df.FTAG > 0)).astype(int)

    base = ["elo_home", "elo_away", "elo_diff", "league"] + [c for c in df.columns if c[:2] in ("h_", "a_") or c.startswith("diff_")]
    odds = ["book_H", "book_D", "book_A", "book_O25"]

    # Step 4: split by time, never at random
    train = df[(df.Season >= FIRST_TRAIN_SEASON) & (df.Season < VALID_SEASON)]
    valid = df[df.Season == VALID_SEASON]
    test = df[df.Season.isin(TEST_SEASONS)]
    print(f"Step 4: learn from {len(train):,} matches (2007/08 to 2021/22), check with {len(valid):,} (2022/23), "
          f"test on {len(test):,} (2023/24 to 2025/26)\n")

    current = {}
    if BACKTEST.exists():
        bt = json.loads(BACKTEST.read_text(encoding="utf-8"))
        current = {"result": bt.get("models", {}).get("Ensemble", {}).get("logloss"),
                   "o25": (bt.get("over25") or {}).get("Dixon-Coles"),
                   "btts": (bt.get("btts") or {}).get("Dixon-Coles")}

    report = []
    for target, label in (("result", "Home/draw/away"), ("o25", "Over 2.5 goals"), ("btts", "Both teams to score")):
        y = f"y_{target}"
        has_odds = test[odds[:3] if target == "result" else ["book_O25"]].notna().all(axis=1)
        print(f"=== {label} ===")
        for version, cols in (("ML without odds", base), ("ML with odds", base + odds)):
            # Step 5: train. Small trees and a slow learning rate stop it memorising noise.
            model = LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=15, min_child_samples=300,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                   verbose=-1)
            model.fit(train[cols], train[y], eval_set=[(valid[cols], valid[y])],
                      callbacks=[early_stopping(100, verbose=False)])
            p = model.predict_proba(test[cols])
            # Step 6: score it
            ll = log_loss(test[y], p)
            acc = accuracy_score(test[y], p.argmax(axis=1))
            print(f"  {version:22s} log loss {ll:.4f}   picks right {acc * 100:.1f}%   ({model.best_iteration_} trees)")
            report.append((label, version, ll))
            if version == "ML with odds" and target == "result":
                imp = pd.Series(model.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
                top = ", ".join(f"{k}" for k in imp.index[:8])
                print(f"    leans on most: {top}")
        # comparisons
        rate = train[y].value_counts(normalize=True).sort_index().values
        print(f"  {'Plain averages':22s} log loss {log_loss(test[y], np.tile(rate, (len(test), 1))):.4f}")
        if current.get(target):
            print(f"  {'Your current model':22s} log loss {current[target]:.4f}   (from data/backtest.json)")
        sub = test[has_odds]
        if target == "result":
            bp = sub[["book_H", "book_D", "book_A"]].values
            print(f"  {'Bookmakers':22s} log loss {log_loss(sub[y], bp):.4f}   picks right {accuracy_score(sub[y], bp.argmax(1)) * 100:.1f}%")
        elif target == "o25":
            bp = sub.book_O25.values
            print(f"  {'Bookmakers':22s} log loss {log_loss(sub[y], np.c_[1 - bp, bp]):.4f}")
        print()

    print("Lower log loss is better. A model is only worth adding to the page if it beats your current model here.")


if __name__ == "__main__":
    main()
