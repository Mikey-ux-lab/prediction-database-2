#!/usr/bin/env python3
"""
Train the prediction models and export them for the calculator page.

    python model.py                  # train, backtest, export
    python model.py --skip-backtest  # faster: reuse the last backtest results

Models
  Dixon-Coles  Attack/defence strength per team, refitted on every run with
               recent matches weighted more heavily (time decay). Gives a full
               scoreline grid, so it covers 1X2, over/under, BTTS, correct
               score, multigoals and combos.
  Elo          A rating per team that moves after every single result.
               Gives home/draw/away only.
  Ensemble     Average of the two for home/draw/away.
  Saves        Same method as Dixon-Coles applied to shots on target;
               expected saves = expected shots on target - expected goals.

Each run also logs predictions for upcoming fixtures and, once results come in,
scores them, so you can see how the model performs on games it had not seen.

National teams get the same two models, trained on every international since
1872 (Elo) and the last eight years (Dixon-Coles). Neutral venues remove home
advantage and friendlies count for less than competitive matches.

Reads:  data/matches.csv, data/fixtures.csv, data/international.csv
Writes: app/model_data.js, data/prediction_log.csv, data/backtest.json
"""
import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import difflib

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MATCHES_FILE = DATA / "matches.csv"
FIXTURES_FILE = DATA / "fixtures.csv"
INTL_FILE = DATA / "international.csv"
MY_FILE = DATA / "my_matches.csv"          # national-team results and fixtures you type in
COUNTRIES_FILE = DATA / "country_names.txt"
CL_FILE = DATA / "champions_league.csv"
BACKTEST_CL_FILE = DATA / "backtest_champions_league.json"
LOG_FILE = DATA / "prediction_log.csv"
BACKTEST_FILE = DATA / "backtest.json"
BACKTEST_INT_FILE = DATA / "backtest_international.json"
APP_DATA = HERE / "app" / "model_data.js"

LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

# ---------------------------------------------------------------- settings
XI = 0.0019               # time decay per day: a match 1 year old counts ~50%
WINDOW_DAYS = 4 * 365     # ignore matches older than this when fitting
RIDGE = 0.5               # light shrinkage of team strengths toward average
PROMOTED_PRIOR = (-0.20, 0.15)      # starting attack/defence for promoted teams
PROMOTED_SOT_PRIOR = (-0.15, 0.10)
PROMOTED_RIDGE = 4.0      # how firmly promoted teams start at that prior
MAX_GOALS = 10            # scoreline grid size used for calculations

ELO_START = 1500
ELO_K = 20
ELO_HFA = 60              # home advantage in Elo points
ELO_REGRESS = 0.15        # pull ratings 15% back to league mean each summer
ELO_PROMOTED_PCTL = 15    # promoted teams start at this percentile of the league

# national teams
INT = "INT"
INT_NAME = "National teams"
INT_XI = 0.0010           # slower decay: countries play far fewer matches
INT_WINDOW_DAYS = 8 * 365
INT_RIDGE = 1.0
INT_FRIENDLY_WEIGHT = 0.6  # a friendly counts 60% of a competitive match
INT_MIN_MATCHES = 10      # matches in the last 4 years needed to be listed
INT_ELO_HFA = 100
INT_BACKTEST_YEARS = 3
LOG_DAYS_AHEAD = 10       # only log predictions for fixtures this close

# Champions League: one strength scale across the five leagues and Europe
CL = "CL"
CL_NAME = "Champions League"
CL_WEIGHT = 3.0           # European matches count extra: they set the level between leagues
EURO_RIDGE = 5.0          # firmer shrinkage: clubs outside the five leagues have few matches
CL_NEW_ELO = 1450         # starting Elo for a club from outside the five leagues
CL_BACKTEST_SEASONS = 3

WARNINGS = []             # problems worth showing on the page (e.g. a misspelt country)
SHEET = {"results": 0, "fixtures": 0, "already_known": 0}   # what your spreadsheet contributed

# Match statistics for league games: key -> (home column, away column, label, usual total line)
STATS = {
    "sh": ("hs", "as", "Shots", 24.5),
    "st": ("hst", "ast", "Shots on target", 8.5),
    "co": ("hc", "ac", "Corners", 9.5),
    "yc": ("hy", "ay", "Yellow cards", 3.5),
    "rc": ("hr", "ar", "Red cards", 0.5),
    "fo": ("hf", "af", "Fouls", 23.5),
}
STATS_BACKTEST_SEASONS = 2

BACKTEST_SEASONS = 3      # walk-forward test on the last N complete seasons
VALUE_EDGE = 0.05         # backtest "value bet" when model prob x odds > 1.05


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- helpers
def season_of(d):
    y = d.year if d.month >= 7 else d.year - 1
    return f"{y}/{(y + 1) % 100:02d}"


def prev_season(s):
    y = int(s[:4]) - 1
    return f"{y}/{(y + 1) % 100:02d}"


def odds(r, *keys):
    for k in keys:
        try:
            v = float(r.get(k) or "")
        except ValueError:
            continue
        if v > 1.0:
            return v
    return math.nan


def num(r, k):
    try:
        return float(r.get(k) or "")
    except ValueError:
        return math.nan


def load_matches():
    if not MATCHES_FILE.exists():
        sys.exit(f"Missing {MATCHES_FILE}. Run: python download_data.py --current")
    rows = []
    with open(MATCHES_FILE, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["Div"] not in LEAGUES:
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["Date"])
    n = len(rows)
    if n == 0:
        sys.exit("matches.csv has no top-5 league matches")
    M = {
        "div": np.array([r["Div"] for r in rows]),
        "season": np.array([r["Season"] for r in rows]),
        "date": np.array([date.fromisoformat(r["Date"]).toordinal() for r in rows]),
        "home": np.array([r["HomeTeam"] for r in rows], dtype=object),
        "away": np.array([r["AwayTeam"] for r in rows], dtype=object),
        "hg": np.array([int(float(r["FTHG"])) for r in rows]),
        "ag": np.array([int(float(r["FTAG"])) for r in rows]),
        "hst": np.array([num(r, "HST") for r in rows]),
        "ast": np.array([num(r, "AST") for r in rows]),
        "hs": np.array([num(r, "HS") for r in rows]), "as": np.array([num(r, "AS") for r in rows]),
        "hc": np.array([num(r, "HC") for r in rows]), "ac": np.array([num(r, "AC") for r in rows]),
        "hy": np.array([num(r, "HY") for r in rows]), "ay": np.array([num(r, "AY") for r in rows]),
        "hr": np.array([num(r, "HR") for r in rows]), "ar": np.array([num(r, "AR") for r in rows]),
        "hf": np.array([num(r, "HF") for r in rows]), "af": np.array([num(r, "AF") for r in rows]),
        # average market odds (falls back to older column names, then Bet365)
        "oH": np.array([odds(r, "AvgH", "BbAvH", "B365H") for r in rows]),
        "oD": np.array([odds(r, "AvgD", "BbAvD", "B365D") for r in rows]),
        "oA": np.array([odds(r, "AvgA", "BbAvA", "B365A") for r in rows]),
        "oO": np.array([odds(r, "Avg>2.5", "BbAv>2.5", "B365>2.5") for r in rows]),
        "oU": np.array([odds(r, "Avg<2.5", "BbAv<2.5", "B365<2.5") for r in rows]),
        # best available odds, used for the value-bet simulation
        "mH": np.array([odds(r, "MaxH", "BbMxH", "B365H") for r in rows]),
        "mD": np.array([odds(r, "MaxD", "BbMxD", "B365D") for r in rows]),
        "mA": np.array([odds(r, "MaxA", "BbMxA", "B365A") for r in rows]),
    }
    M["res"] = np.where(M["hg"] > M["ag"], 0, np.where(M["hg"] == M["ag"], 1, 2))
    M["n"] = n
    return M


def load_fixtures(M):
    if not FIXTURES_FILE.exists():
        return []
    played = set(zip(M["div"], M["home"], M["away"], M["date"]))
    today = date.today().toordinal()
    out = []
    with open(FIXTURES_FILE, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Div") not in LEAGUES and r.get("Div") != CL:
                continue
            try:
                d = date.fromisoformat(r["Date"])
            except (ValueError, KeyError):
                continue
            if d.toordinal() < today:
                continue
            if (r["Div"], r["HomeTeam"], r["AwayTeam"], d.toordinal()) in played:
                continue
            fx = {"div": r["Div"], "date": r["Date"], "time": r.get("Time", ""),
                  "tz": r.get("TZ") or "UK", "home": r["HomeTeam"], "away": r["AwayTeam"],
                  "odds": None, "best": None}
            avg = {"H": odds(r, "AvgH", "B365H"), "D": odds(r, "AvgD", "B365D"), "A": odds(r, "AvgA", "B365A"),
                   "O25": odds(r, "Avg>2.5", "B365>2.5"), "U25": odds(r, "Avg<2.5", "B365<2.5")}
            if any(not math.isnan(v) for v in avg.values()):
                fx["odds"] = avg
                fx["best"] = {"H": odds(r, "MaxH", "B365H"), "D": odds(r, "MaxD", "B365D"),
                              "A": odds(r, "MaxA", "B365A"),
                              "O25": odds(r, "Max>2.5", "B365>2.5"), "U25": odds(r, "Max<2.5", "B365<2.5")}
            if r["Div"] == CL:
                fx["comp"] = CL_NAME + (", " + r["Comp"] if r.get("Comp") else "")
                fx["neutral"] = r.get("Neutral") == "TRUE"
            out.append(fx)
    return out


# ---------------------------------------------------------------- Dixon-Coles
def _nll(p, n, h, a, x, y, w, hm, masks, use_rho, prior_att, prior_def, ridge):
    att, de = p[:n], p[n:2 * n]
    mu0, hfa = p[2 * n], p[2 * n + 1]
    r = p[2 * n + 2] if use_rho else 0.0
    ll = mu0 + hfa * hm + att[h] + de[a]   # hm is 0 at a neutral venue
    lm = mu0 + att[a] + de[h]
    lam, mu = np.exp(ll), np.exp(lm)
    f = x * ll - lam + y * lm - mu
    gl = x - lam
    gm = y - mu
    gr = 0.0
    if use_rho:
        m00, m01, m10, m11 = masks
        tau = np.ones_like(lam)
        lm00 = lam[m00] * mu[m00]
        tau[m00] = 1 - lm00 * r
        tau[m01] = 1 + lam[m01] * r
        tau[m10] = 1 + mu[m10] * r
        tau[m11] = 1 - r
        tau = np.maximum(tau, 1e-9)
        f = f + np.log(tau)
        gl[m00] -= lm00 * r / tau[m00]
        gm[m00] -= lm00 * r / tau[m00]
        gl[m01] += lam[m01] * r / tau[m01]
        gm[m10] += mu[m10] * r / tau[m10]
        gr = (np.sum(w[m00] * -lm00 / tau[m00]) + np.sum(w[m01] * lam[m01] / tau[m01])
              + np.sum(w[m10] * mu[m10] / tau[m10]) - np.sum(w[m11] / tau[m11]))
    gl = w * gl
    gm = w * gm
    da, dd = att - prior_att, de - prior_def
    val = -np.sum(w * f) + np.sum(ridge * (da ** 2 + dd ** 2))
    g_att = -(np.bincount(h, gl, n) + np.bincount(a, gm, n)) + 2 * ridge * da
    g_def = -(np.bincount(a, gl, n) + np.bincount(h, gm, n)) + 2 * ridge * dd
    grad = np.concatenate([g_att, g_def, [-(gl.sum() + gm.sum()), -np.sum(gl * hm)]])
    if use_rho:
        grad = np.append(grad, -gr)
    return val, grad


class PoissonFit:
    """Fitted attack/defence model (Dixon-Coles when use_rho=True)."""

    def __init__(self, teams, att, de, mu0, hfa, rho):
        self.teams = teams
        self.idx = {t: i for i, t in enumerate(teams)}
        self.att, self.de, self.mu0, self.hfa, self.rho = att, de, mu0, hfa, rho

    def rates(self, home, away, neutral=False):
        i, j = self.idx[home], self.idx[away]
        lam = math.exp(self.mu0 + (0.0 if neutral else self.hfa) + self.att[i] + self.de[j])
        mu = math.exp(self.mu0 + self.att[j] + self.de[i])
        return lam, mu

    def team_dict(self):
        return {t: (float(self.att[i]), float(self.de[i])) for i, t in enumerate(self.teams)}


def fit_poisson(home, away, x, y, w, promoted=(), extra_teams=(), use_rho=True,
                prior=PROMOTED_PRIOR, init=None, hm=None, ridge_base=RIDGE):
    teams = sorted(set(home) | set(away) | set(extra_teams) | set(promoted))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h = np.array([idx[t] for t in home], dtype=np.int64)
    a = np.array([idx[t] for t in away], dtype=np.int64)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    prior_att = np.array([prior[0] if t in promoted else 0.0 for t in teams])
    prior_def = np.array([prior[1] if t in promoted else 0.0 for t in teams])
    ridge = np.array([PROMOTED_RIDGE if t in promoted else ridge_base for t in teams])
    hm = np.ones(len(x)) if hm is None else np.asarray(hm, dtype=float)
    masks = ((x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1))

    p0 = np.concatenate([prior_att, prior_def, [math.log(max(np.average(x + y, weights=w) / 2, 0.1)), 0.25]])
    if init is not None:
        old_t, old_mu0, old_hfa = init.team_dict(), init.mu0, init.hfa
        for t, i in idx.items():
            if t in old_t and t not in promoted:
                p0[i], p0[n + i] = old_t[t]
        p0[2 * n], p0[2 * n + 1] = old_mu0, old_hfa
    bounds = [(None, None)] * (2 * n + 2)
    if use_rho:
        p0 = np.append(p0, init.rho if init is not None else -0.05)
        bounds.append((-0.25, 0.25))
    res = minimize(_nll, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                   args=(n, h, a, x, y, w, hm, masks, use_rho, prior_att, prior_def, ridge),
                   options={"maxiter": 500})
    p = res.x
    return PoissonFit(teams, p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1],
                      p[2 * n + 2] if use_rho else 0.0)


def score_grid(lam, mu, rho, maxg=MAX_GOALS):
    i = np.arange(maxg + 1)
    g = np.outer(poisson.pmf(i, lam), poisson.pmf(i, mu))
    g[0, 0] *= 1 - lam * mu * rho
    g[0, 1] *= 1 + lam * rho
    g[1, 0] *= 1 + mu * rho
    g[1, 1] *= 1 - rho
    g = np.clip(g, 0, None)
    return g / g.sum()


def grid_markets(g):
    n = g.shape[0]
    tot = np.add.outer(np.arange(n), np.arange(n))
    return {
        "H": float(np.tril(g, -1).sum()), "D": float(np.trace(g)), "A": float(np.triu(g, 1).sum()),
        "O25": float(g[tot > 2].sum()), "BTTS": float(g[1:, 1:].sum()),
    }


def fit_window(M, league, T, promoted=(), extra_teams=(), init=None, what="goals"):
    """Fit on one league's matches strictly before day T inside the window."""
    m = (M["div"] == league) & (M["date"] < T) & (M["date"] >= T - WINDOW_DAYS)
    if what == "goals":
        x, y, rho, prior = M["hg"][m], M["ag"][m], True, PROMOTED_PRIOR
    else:
        m = m & np.isfinite(M["hst"]) & np.isfinite(M["ast"])
        x, y, rho, prior = M["hst"][m], M["ast"][m], False, PROMOTED_SOT_PRIOR
    if m.sum() < 50:
        return None
    w = np.exp(-XI * (T - M["date"][m]))
    return fit_poisson(M["home"][m], M["away"][m], x, y, w, promoted=set(promoted),
                       extra_teams=extra_teams, use_rho=rho, prior=prior, init=init)


def fit_stat(M, league, T, key, promoted=(), extra_teams=(), init=None):
    """Attack/defence-style fit for one match statistic, plus how spread out it is."""
    hc, ac = STATS[key][0], STATS[key][1]
    m = ((M["div"] == league) & (M["date"] < T) & (M["date"] >= T - WINDOW_DAYS)
         & np.isfinite(M[hc]) & np.isfinite(M[ac]))
    if m.sum() < 50:
        return None, 0.0
    w = np.exp(-XI * (T - M["date"][m]))
    prior = PROMOTED_SOT_PRIOR if key in ("sh", "st", "co") else (0.0, 0.0)
    fit = fit_poisson(M["home"][m], M["away"][m], M[hc][m], M[ac][m], w, promoted=set(promoted),
                      extra_teams=extra_teams, use_rho=False, prior=prior, init=init)
    h = np.array([fit.idx[t] for t in M["home"][m]])
    a = np.array([fit.idx[t] for t in M["away"][m]])
    lam = np.exp(fit.mu0 + fit.hfa + fit.att[h] + fit.de[a])
    mu = np.exp(fit.mu0 + fit.att[a] + fit.de[h])
    # counts vary more than a Poisson allows; alpha measures the extra spread (0 = Poisson)
    num_ = np.sum(w * ((M[hc][m] - lam) ** 2 - lam)) + np.sum(w * ((M[ac][m] - mu) ** 2 - mu))
    alpha = max(0.0, float(num_ / (np.sum(w * lam ** 2) + np.sum(w * mu ** 2))))
    return fit, alpha


def nb_pmf(mean, alpha, n=80):
    """Negative binomial probabilities for 0..n (Poisson when alpha is 0)."""
    p = np.zeros(n + 1)
    if alpha < 1e-6:
        p[0] = math.exp(-mean)
        for i in range(n):
            p[i + 1] = p[i] * mean / (i + 1)
    else:
        k = 1 / alpha
        p[0] = (k / (k + mean)) ** k
        for i in range(n):
            p[i + 1] = p[i] * (i + k) / (i + 1) * mean / (k + mean)
    return p / p.sum()


def prob_over(line, alpha, *means):
    """Chance that the sum of independent counts with these means beats the line."""
    pmf = nb_pmf(means[0], alpha)
    for m in means[1:]:
        pmf = np.convolve(pmf, nb_pmf(m, alpha))
    return float(pmf[int(math.floor(line)) + 1:].sum())


def backtest_stats(M):
    """Walk-forward test of the match-statistics models against the plain league average."""
    cur = season_of(date.today())
    seasons = sorted(s for s in set(M["season"]) if s < cur)[-STATS_BACKTEST_SEASONS:]
    out = {}
    for key, (hc, ac, label, line) in STATS.items():
        p_model, hit, tot_pred, tot_real, tot_avg = [], [], [], [], []
        for L in LEAGUES:
            idx = np.where((M["div"] == L) & np.isin(M["season"], seasons) & np.isfinite(M[hc]) & np.isfinite(M[ac]))[0]
            if len(idx) == 0:
                continue
            months = np.array([(lambda d: d.year * 12 + d.month - 1)(date.fromordinal(int(x))) for x in M["date"][idx]])
            prev = None
            for mo in np.unique(months):
                T = date(mo // 12, mo % 12 + 1, 1).toordinal()
                block = idx[months == mo]
                teams = set(M["home"][block]) | set(M["away"][block])
                fit, alpha = fit_stat(M, L, T, key, extra_teams=teams, init=prev)
                if fit is None:
                    continue
                prev = fit
                past = (M["div"] == L) & (M["date"] < T) & (M["date"] >= T - 365) & np.isfinite(M[hc]) & np.isfinite(M[ac])
                avg = float(np.mean(M[hc][past] + M[ac][past]))
                for i in block:
                    lam, mu = fit.rates(M["home"][i], M["away"][i])
                    real = M[hc][i] + M[ac][i]
                    p_model.append(prob_over(line, alpha, lam, mu))
                    hit.append(1.0 if real > line else 0.0)
                    tot_pred.append(lam + mu)
                    tot_real.append(real)
                    tot_avg.append(avg)
        if not hit:
            continue
        hit, p_model = np.array(hit), np.array(p_model)
        out[key] = {"label": label, "line": line, "matches": int(len(hit)), "over_rate": float(hit.mean()),
                    "model": binary_logloss(p_model, hit),
                    "base": binary_logloss(np.full(len(hit), hit.mean()), hit),
                    "mae_model": float(np.mean(np.abs(np.array(tot_pred) - np.array(tot_real)))),
                    "mae_average": float(np.mean(np.abs(np.array(tot_avg) - np.array(tot_real))))}
    return {"seasons": seasons, "stats": out}


def promoted_teams(M, league, season, current_teams):
    prev = set(M["home"][(M["div"] == league) & (M["season"] == prev_season(season))])
    return set(current_teams) - prev if prev else set()


# ---------------------------------------------------------------- Elo
class Elo:
    def __init__(self):
        self.r = {}                       # (league, team) -> rating
        self.season_teams = defaultdict(set)
        self.league_season = {}

    def _start_season(self, L, s):
        prev = self.season_teams.get((L, prev_season(s)), set())
        vals = [self.r[(L, t)] for t in prev if (L, t) in self.r]
        if vals:
            mean = float(np.mean(vals))
            for t in prev:
                self.r[(L, t)] = mean + (1 - ELO_REGRESS) * (self.r[(L, t)] - mean)
        self.league_season[L] = s

    def _enter(self, L, s, t):
        if t in self.season_teams[(L, s)]:
            return
        prev = self.season_teams.get((L, prev_season(s)), set())
        if t not in prev or (L, t) not in self.r:
            vals = [self.r[(L, u)] for u in prev if (L, u) in self.r]
            self.r[(L, t)] = float(np.percentile(vals, ELO_PROMOTED_PCTL)) if vals else ELO_START
        self.season_teams[(L, s)].add(t)

    def prepare(self, L, s, t):
        if self.league_season.get(L) != s:
            self._start_season(L, s)
        self._enter(L, s, t)

    def diff(self, L, home, away):
        return self.r[(L, home)] + ELO_HFA - self.r[(L, away)]

    def update(self, L, home, away, hg, ag):
        d = self.diff(L, home, away)
        e = 1 / (1 + 10 ** (-d / 400))
        s = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag)
        g = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = ELO_K * g * (s - e)
        self.r[(L, home)] += delta
        self.r[(L, away)] -= delta

    def rating_for(self, L, s, t):
        """Rating a team would have in season s (handles not-yet-started seasons)."""
        if t in self.season_teams.get((L, s), set()):
            return self.r[(L, t)]
        if self.league_season.get(L) == s:
            prev = self.season_teams.get((L, prev_season(s)), set())
            vals = [self.r[(L, u)] for u in prev if (L, u) in self.r]
        else:  # season has not begun in the data yet
            last = self.league_season.get(L)
            teams = self.season_teams.get((L, last), set())
            vals = [self.r[(L, u)] for u in teams]
            if t in teams:
                mean = float(np.mean(vals))
                return mean + (1 - ELO_REGRESS) * (self.r[(L, t)] - mean)
        return float(np.percentile(vals, ELO_PROMOTED_PCTL)) if vals else ELO_START


def run_elo(M):
    elo = Elo()
    pre = np.zeros(M["n"])
    for i in range(M["n"]):
        L, s, h, a = M["div"][i], M["season"][i], M["home"][i], M["away"][i]
        elo.prepare(L, s, h)
        elo.prepare(L, s, a)
        pre[i] = elo.diff(L, h, a)
        elo.update(L, h, a, M["hg"][i], M["ag"][i])
    return elo, pre


def _sig(z):
    return 1 / (1 + np.exp(-z))


def logit_probs(diff, c):
    """Ordered logit: Elo difference -> home/draw/away probabilities."""
    c1, k, b = c
    z = np.asarray(diff, dtype=float) / 100 * b
    pa = _sig(c1 - z)
    pad = _sig(c1 + math.exp(k) - z)
    return np.stack([1 - pad, pad - pa, pa], axis=-1)


def fit_logit(diff, res):
    def nll(c):
        P = np.clip(logit_probs(diff, c), 1e-12, 1)
        return -np.mean(np.log(P[np.arange(len(res)), res]))
    out = minimize(nll, [-0.8, 0.0, 0.5], method="Nelder-Mead", options={"maxiter": 2000})
    return [float(v) for v in out.x]


# ---------------------------------------------------------------- backtest
def metrics(P, y):
    P = np.clip(P, 1e-12, 1)
    oh = np.eye(P.shape[1])[y]
    return {
        "logloss": float(-np.mean(np.log(P[np.arange(len(y)), y]))),
        "brier": float(np.mean(np.sum((P - oh) ** 2, axis=1))),
        "accuracy": float(np.mean(P.argmax(1) == y)),
    }


def binary_logloss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def value_roi(P, y, best):
    stake = profit = 0.0
    for k in range(3):
        o = best[:, k]
        bet = np.isfinite(o) & (P[:, k] * o > 1 + VALUE_EDGE)
        stake += bet.sum()
        profit += np.sum(np.where(y[bet] == k, o[bet] - 1, -1))
    return {"bets": int(stake), "roi": float(profit / stake) if stake else None}


def backtest(M, elo_pre):
    cur = season_of(date.today())
    seasons = sorted(s for s in set(M["season"]) if s < cur)
    if len(seasons) < BACKTEST_SEASONS + 3:
        log("Not enough seasons for a backtest; skipping")
        return None
    bt = seasons[-BACKTEST_SEASONS:]
    bt_mask = np.isin(M["season"], bt)
    start = M["date"][bt_mask].min()
    burn_in = M["date"][M["season"] == seasons[2]].min()
    train = (M["date"] >= burn_in) & (M["date"] < start)
    logit_c = fit_logit(elo_pre[train], M["res"][train])

    idx_all, dc_P, dc_O, dc_B = [], [], [], []
    t0 = time.time()
    for L in LEAGUES:
        idxL = np.where(bt_mask & (M["div"] == L))[0]
        if len(idxL) == 0:
            continue
        weeks = M["date"][idxL] - (M["date"][idxL] % 7)
        prev_fit = None
        for wk in np.unique(weeks):
            test = idxL[weeks == wk]
            s = M["season"][test[0]]
            cur_teams = set(M["home"][(M["div"] == L) & (M["season"] == s) & (M["date"] < wk)])
            cur_teams |= set(M["home"][test]) | set(M["away"][test])
            promo = promoted_teams(M, L, s, cur_teams)
            fit = fit_window(M, L, wk, promoted=promo, extra_teams=cur_teams, init=prev_fit)
            if fit is None:
                continue
            prev_fit = fit
            for i in test:
                g = score_grid(*fit.rates(M["home"][i], M["away"][i]), fit.rho)
                mk = grid_markets(g)
                idx_all.append(i)
                dc_P.append([mk["H"], mk["D"], mk["A"]])
                dc_O.append(mk["O25"])
                dc_B.append(mk["BTTS"])
        log(f"  backtest {LEAGUES[L]}: done ({time.time() - t0:.0f}s)")

    I = np.array(idx_all)
    y = M["res"][I]
    dcP = np.array(dc_P)
    eloP = logit_probs(elo_pre[I], logit_c)
    ensP = (dcP + eloP) / 2
    book = np.stack([1 / M["oH"][I], 1 / M["oD"][I], 1 / M["oA"][I]], axis=1)
    has = np.all(np.isfinite(book), axis=1)
    bookP = book / book.sum(axis=1, keepdims=True)
    best = np.stack([M["mH"][I], M["mD"][I], M["mA"][I]], axis=1)

    out = {"seasons": bt, "matches": int(len(I)), "with_odds": int(has.sum()),
           "models": {}, "by_league": {}, "value_bets": {}, "calibration": None, "elo_logit": logit_c}
    for name, P in (("Dixon-Coles", dcP), ("Elo", eloP), ("Ensemble", ensP)):
        out["models"][name] = metrics(P[has], y[has])
        out["value_bets"][name] = value_roi(P[has], y[has], best[has])
    out["models"]["Bookmakers"] = metrics(bookP[has], y[has])
    out["home_win_rate"] = float(np.mean(y[has] == 0))  # accuracy of always picking home

    # Over/under 2.5 and BTTS
    goals = M["hg"][I] + M["ag"][I]
    over = (goals > 2).astype(float)
    btts = ((M["hg"][I] > 0) & (M["ag"][I] > 0)).astype(float)
    dcO, dcB = np.array(dc_O), np.array(dc_B)
    ou_book = np.stack([1 / M["oO"][I], 1 / M["oU"][I]], axis=1)
    has_ou = np.all(np.isfinite(ou_book), axis=1)
    book_over = ou_book[:, 0] / ou_book.sum(axis=1)
    out["over25"] = {
        "Dixon-Coles": binary_logloss(dcO[has_ou], over[has_ou]),
        "Bookmakers": binary_logloss(book_over[has_ou], over[has_ou]),
        "Base rate": binary_logloss(np.full(int(has_ou.sum()), over[has_ou].mean()), over[has_ou]),
        "matches": int(has_ou.sum()),
    }
    out["btts"] = {
        "Dixon-Coles": binary_logloss(dcB, btts),
        "Base rate": binary_logloss(np.full(len(btts), btts.mean()), btts),
        "matches": int(len(btts)),
    }
    for L in LEAGUES:
        m = has & (M["div"][I] == L)
        if m.sum() == 0:
            continue
        out["by_league"][L] = {name: metrics(P[m], y[m])["logloss"] for name, P in
                               (("Dixon-Coles", dcP), ("Elo", eloP), ("Ensemble", ensP), ("Bookmakers", bookP))}
    # calibration of the ensemble across all three outcomes
    pp = ensP.ravel()
    hit = np.eye(3)[y].ravel()
    bins = np.clip((pp * 10).astype(int), 0, 9)
    out["calibration"] = [
        {"bin": f"{b * 10}-{b * 10 + 10}%", "predicted": float(pp[bins == b].mean()),
         "actual": float(hit[bins == b].mean()), "n": int((bins == b).sum())}
        for b in range(10) if (bins == b).sum() >= 30
    ]
    return out


# ---------------------------------------------------------------- national teams
INT_MAJOR = {"UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup",
             "Confederations Cup", "Oceania Nations Cup", "CONCACAF Championship"}


def int_k(tournament):
    """How much one result moves a national team's Elo rating."""
    if tournament == "FIFA World Cup":
        return 60.0
    if tournament in INT_MAJOR:
        return 50.0
    if "qualification" in tournament or "Nations League" in tournament:
        return 40.0
    if tournament == "Friendly":
        return 20.0
    return 30.0


MY_HEADER = "date,home_team,away_team,home_score,away_score,tournament,neutral"
MY_ALIASES = {
    "cote d'ivoire": "Ivory Coast", "côte d'ivoire": "Ivory Coast", "cote divoire": "Ivory Coast",
    "usa": "United States", "us": "United States", "korea republic": "South Korea", "korea dpr": "North Korea",
    "congo dr": "DR Congo", "democratic republic of congo": "DR Congo", "drc": "DR Congo",
    "turkiye": "Turkey", "türkiye": "Turkey", "czechia": "Czech Republic", "cabo verde": "Cape Verde",
    "uae": "United Arab Emirates", "ir iran": "Iran", "china pr": "China", "ireland": "Republic of Ireland",
    "bosnia": "Bosnia and Herzegovina", "bosnia & herzegovina": "Bosnia and Herzegovina",
    "macedonia": "North Macedonia", "swaziland": "Eswatini", "holland": "Netherlands",
    "the gambia": "Gambia", "sao tome": "São Tomé and Príncipe", "curacao": "Curaçao",
}


def _my_date(s):
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def read_my_matches(names, existing):
    """Results and fixtures you typed into data/my_matches.csv."""
    if not MY_FILE.exists():
        MY_FILE.write_text(MY_HEADER + "\n", encoding="utf-8")
        return [], []
    text = MY_FILE.read_text(encoding="utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text.strip() else ""
    delim = ";" if first.count(";") > first.count(",") else ","   # Excel uses ; in some regions
    have = defaultdict(set)
    for d, h, a, *_ in existing:
        have[(h, a)].add(date.fromisoformat(d).toordinal())
    lower = {n.lower(): n for n in names}
    for alt, real in MY_ALIASES.items():   # other common spellings
        if real in names:
            lower.setdefault(alt, real)
    results, fixtures = [], []
    today = date.today()
    for n, r in enumerate(csv.DictReader(text.splitlines(), delimiter=delim), start=2):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        if not any(r.values()):
            continue
        d = _my_date(r.get("date"))
        if d is None:
            WARNINGS.append(f"my_matches.csv row {n}: the date '{r.get('date', '')}' isn't readable. Use 2026-10-03 or 03/10/2026.")
            continue
        pair = []
        for col in ("home_team", "away_team"):
            raw = r.get(col, "")
            name = lower.get(raw.lower())
            if name is None:
                close = difflib.get_close_matches(raw, sorted(names), n=1, cutoff=0.6)
                hint = f" Did you mean '{close[0]}'?" if close else " See data/country_names.txt for the spellings."
                WARNINGS.append(f"my_matches.csv row {n}: '{raw}' isn't a team name the model knows.{hint}")
            pair.append(name)
        if None in pair or pair[0] == pair[1]:
            continue
        h, a = pair
        neutral = r.get("neutral", "").lower() in ("true", "yes", "y", "1", "x")
        tour = r.get("tournament") or "Friendly"
        hs, as_ = r.get("home_score", ""), r.get("away_score", "")
        if hs == "" and as_ == "":
            if d >= today:
                fixtures.append({"date": d.isoformat(), "home_team": h, "away_team": a,
                                 "tournament": tour, "neutral": "TRUE" if neutral else "FALSE"})
            else:
                WARNINGS.append(f"my_matches.csv row {n}: {h} v {a} on {d.isoformat()} is in the past but has no score. Type the score in, or delete the row.")
            continue
        try:
            hg, ag = int(float(hs)), int(float(as_))
        except ValueError:
            WARNINGS.append(f"my_matches.csv row {n}: the score '{hs}-{as_}' isn't two whole numbers.")
            continue
        if any(abs(d.toordinal() - x) <= 2 for x in have[(h, a)]):
            SHEET["already_known"] += 1
            continue  # the main dataset has caught up with this match
        have[(h, a)].add(d.toordinal())
        results.append((d.isoformat(), h, a, hg, ag, tour, neutral))
    SHEET["results"], SHEET["fixtures"] = len(results), len(fixtures)
    log(f"Your spreadsheet: {len(results)} new results, {len(fixtures)} upcoming fixtures, "
        f"{SHEET['already_known']} already in the main data")
    return results, fixtures


def load_international():
    if not INTL_FILE.exists():
        return None
    rows, upcoming = [], []
    today = date.today().isoformat()
    with open(INTL_FILE, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                hg, ag = int(r["home_score"]), int(r["away_score"])
            except (ValueError, KeyError, TypeError):
                if (r.get("date") or "") >= today:
                    upcoming.append(r)
                continue
            rows.append((r["date"], r["home_team"], r["away_team"], hg, ag, r["tournament"],
                         (r.get("neutral") or "").strip().upper() == "TRUE"))
    # FIFA-level teams only: anyone who has played a World Cup qualifier since 2000
    fifa = set()
    for d, h, a, _, _, t, _ in rows:
        if t == "FIFA World Cup qualification" and d >= "2000":
            fifa.update((h, a))
    COUNTRIES_FILE.write_text("\n".join(sorted(fifa)) + "\n", encoding="utf-8")
    mine_res, mine_fix = read_my_matches(fifa, rows)
    rows += mine_res
    upcoming += mine_fix
    rows = sorted((r for r in rows if r[1] in fifa and r[2] in fifa), key=lambda r: r[0])
    if not rows:
        return None
    I = {
        "date": np.array([date.fromisoformat(r[0]).toordinal() for r in rows]),
        "home": np.array([r[1] for r in rows], dtype=object),
        "away": np.array([r[2] for r in rows], dtype=object),
        "hg": np.array([r[3] for r in rows]), "ag": np.array([r[4] for r in rows]),
        "tour": np.array([r[5] for r in rows], dtype=object),
        "neutral": np.array([r[6] for r in rows], dtype=bool),
        "n": len(rows),
        "upcoming": [u for u in upcoming if u.get("home_team") in fifa and u.get("away_team") in fifa],
    }
    I["res"] = np.where(I["hg"] > I["ag"], 0, np.where(I["hg"] == I["ag"], 1, 2))
    return I


def run_elo_int(I):
    r = defaultdict(lambda: float(ELO_START))
    pre = np.zeros(I["n"])
    for i in range(I["n"]):
        h, a = I["home"][i], I["away"][i]
        d = r[h] + (0.0 if I["neutral"][i] else INT_ELO_HFA) - r[a]
        pre[i] = d
        e = 1 / (1 + 10 ** (-d / 400))
        hg, ag = I["hg"][i], I["ag"][i]
        sc = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag)
        g = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = int_k(I["tour"][i]) * g * (sc - e)
        r[h] += delta
        r[a] -= delta
    return dict(r), pre


def fit_int(I, T, extra_teams=(), init=None):
    m = (I["date"] < T) & (I["date"] >= T - INT_WINDOW_DAYS)
    w = np.exp(-INT_XI * (T - I["date"][m])) * np.where(I["tour"][m] == "Friendly", INT_FRIENDLY_WEIGHT, 1.0)
    fit = fit_poisson(I["home"][m], I["away"][m], I["hg"][m], I["ag"][m], w, extra_teams=extra_teams,
                      init=init, hm=(~I["neutral"][m]).astype(float), ridge_base=INT_RIDGE)
    return fit, m


def backtest_int(I, elo_pre):
    end = int(I["date"].max())
    start = end - INT_BACKTEST_YEARS * 365
    train = (I["date"] >= date(2006, 1, 1).toordinal()) & (I["date"] < start)
    logit_c = fit_logit(elo_pre[train], I["res"][train])
    test = np.where(I["date"] >= start)[0]
    months = np.array([(lambda d: d.year * 12 + d.month - 1)(date.fromordinal(int(x))) for x in I["date"][test]])
    idx, dc_P, dc_O, dc_B = [], [], [], []
    prev = None
    for mo in np.unique(months):
        T = date(mo // 12, mo % 12 + 1, 1).toordinal()
        block = test[months == mo]
        fit, m = fit_int(I, T, extra_teams=set(I["home"][block]) | set(I["away"][block]), init=prev)
        prev = fit
        seen = defaultdict(int)
        for t in np.concatenate([I["home"][m], I["away"][m]]):
            seen[t] += 1
        for i in block:
            h, a = I["home"][i], I["away"][i]
            if seen[h] < 5 or seen[a] < 5:
                continue  # not enough history to judge the model on
            mk = grid_markets(score_grid(*fit.rates(h, a, bool(I["neutral"][i])), fit.rho))
            idx.append(i)
            dc_P.append([mk["H"], mk["D"], mk["A"]])
            dc_O.append(mk["O25"])
            dc_B.append(mk["BTTS"])
    J = np.array(idx)
    y = I["res"][J]
    dcP = np.array(dc_P)
    eloP = logit_probs(elo_pre[J], logit_c)
    ensP = (dcP + eloP) / 2
    comp = I["tour"][J] != "Friendly"
    out = {"from": date.fromordinal(start).isoformat(), "to": date.fromordinal(end).isoformat(),
           "matches": int(len(J)), "competitive": int(comp.sum()), "elo_logit": logit_c,
           "home_win_rate": float(np.mean(y == 0)), "models": {}, "competitive_only": {}, "friendlies_only": {}}
    for name, P in (("Dixon-Coles", dcP), ("Elo", eloP), ("Ensemble", ensP)):
        out["models"][name] = metrics(P, y)
        out["competitive_only"][name] = metrics(P[comp], y[comp])
        out["friendlies_only"][name] = metrics(P[~comp], y[~comp])
    goals = I["hg"][J] + I["ag"][J]
    over = (goals > 2).astype(float)
    btts = ((I["hg"][J] > 0) & (I["ag"][J] > 0)).astype(float)
    out["over25"] = {"Dixon-Coles": binary_logloss(np.array(dc_O), over),
                     "Base rate": binary_logloss(np.full(len(over), over.mean()), over)}
    out["btts"] = {"Dixon-Coles": binary_logloss(np.array(dc_B), btts),
                   "Base rate": binary_logloss(np.full(len(btts), btts.mean()), btts)}
    pp, hit = ensP.ravel(), np.eye(3)[y].ravel()
    bins = np.clip((pp * 10).astype(int), 0, 9)
    out["calibration"] = [
        {"bin": f"{b * 10}-{b * 10 + 10}%", "predicted": float(pp[bins == b].mean()),
         "actual": float(hit[bins == b].mean()), "n": int((bins == b).sum())}
        for b in range(10) if (bins == b).sum() >= 30
    ]
    return out


def form_int(I, team, k=5):
    idx = np.where((I["home"] == team) | (I["away"] == team))[0][-k:]
    out = []
    for i in idx:
        home = I["home"][i] == team
        gf, ga = (I["hg"][i], I["ag"][i]) if home else (I["ag"][i], I["hg"][i])
        out.append({"r": "W" if gf > ga else "D" if gf == ga else "L", "s": f"{gf}-{ga}",
                    "opp": I["away"][i] if home else I["home"][i],
                    "ha": "N" if I["neutral"][i] else "H" if home else "A",
                    "d": date.fromordinal(int(I["date"][i])).isoformat(), "c": I["tour"][i]})
    return out


def final_int(I, ratings, logit_c):
    T = date.today().toordinal() + 1
    fit, _ = fit_int(I, T)
    recent = I["date"] >= T - 4 * 365
    seen = defaultdict(int)
    for t in np.concatenate([I["home"][recent], I["away"][recent]]):
        seen[t] += 1
    teams = {}
    for t in sorted(t for t, c in seen.items() if c >= INT_MIN_MATCHES and t in fit.idx):
        i = fit.idx[t]
        teams[t] = {"att": round(float(fit.att[i]), 4), "def": round(float(fit.de[i]), 4),
                    "elo": round(ratings.get(t, ELO_START), 1), "promoted": False, "form": form_int(I, t)}
    log(f"  {INT_NAME}: {len(teams)} teams, home adv x{math.exp(fit.hfa):.2f}, rho {fit.rho:+.3f}")
    return {"name": INT_NAME, "international": True, "neutral_option": True, "mu0": round(float(fit.mu0), 4),
            "hfa": round(float(fit.hfa), 4), "rho": round(float(fit.rho), 4), "sot": None,
            "elo": {"hfa": INT_ELO_HFA, "logit": logit_c}, "teams": teams}


# ---------------------------------------------------------------- Champions League
def load_cl():
    """Bundled history plus any played matches from this season's live schedule."""
    rows, seen = [], set()
    if CL_FILE.exists():
        with open(CL_FILE, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows.append((r["Date"], r["HomeTeam"], r["AwayTeam"], int(r["FTHG"]), int(r["FTAG"]),
                             r["Neutral"] == "TRUE", r["Season"], r["Stage"]))
                seen.add((r["Date"], r["HomeTeam"], r["AwayTeam"]))
    try:
        from download_data import load_schedule
        for m in load_schedule():
            if m["div"] != CL or m["score"] is None or (m["date"], m["home"], m["away"]) in seen:
                continue
            rows.append((m["date"], m["home"], m["away"], m["score"][0], m["score"][1],
                         m["neutral"], m["season"], m["stage"]))
    except Exception as e:  # the schedule is optional
        log(f"  (no live Champions League schedule: {e})")
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    C = {"date": np.array([date.fromisoformat(r[0]).toordinal() for r in rows]),
         "home": np.array([r[1] for r in rows], dtype=object), "away": np.array([r[2] for r in rows], dtype=object),
         "hg": np.array([r[3] for r in rows]), "ag": np.array([r[4] for r in rows]),
         "neutral": np.array([r[5] for r in rows], dtype=bool),
         "season": np.array([r[6] for r in rows]), "stage": np.array([r[7] for r in rows], dtype=object),
         "n": len(rows)}
    C["res"] = np.where(C["hg"] > C["ag"], 0, np.where(C["hg"] == C["ag"], 1, 2))
    return C


def run_elo_euro(M, C):
    """One Elo pool for the five leagues and the Champions League."""
    r, league_season, teams_in = {}, {}, defaultdict(set)
    events = sorted([(int(M["date"][i]), 0, i) for i in range(M["n"])]
                    + [(int(C["date"][j]), 1, j) for j in range(C["n"])])
    pre = np.zeros(C["n"])
    for _, kind, i in events:
        if kind == 0:
            L, s, h, a, hg, ag = M["div"][i], M["season"][i], M["home"][i], M["away"][i], M["hg"][i], M["ag"][i]
            prev = teams_in.get((L, prev_season(s)), set())
            if league_season.get(L) != s:
                vals = [r[t] for t in prev if t in r]
                if vals:
                    mean = float(np.mean(vals))
                    for t in prev:
                        r[t] = mean + (1 - ELO_REGRESS) * (r[t] - mean)
                league_season[L] = s
            for t in (h, a):
                if t not in teams_in[(L, s)]:
                    if t not in prev:  # promoted (or the very first season)
                        vals = [r[u] for u in prev if u in r]
                        r[t] = float(np.percentile(vals, ELO_PROMOTED_PCTL)) if vals else ELO_START
                    teams_in[(L, s)].add(t)
            d = r[h] + ELO_HFA - r[a]
        else:
            h, a, hg, ag = C["home"][i], C["away"][i], C["hg"][i], C["ag"][i]
            r.setdefault(h, CL_NEW_ELO)
            r.setdefault(a, CL_NEW_ELO)
            d = r[h] + (0.0 if C["neutral"][i] else ELO_HFA) - r[a]
            pre[i] = d
        e = 1 / (1 + 10 ** (-d / 400))
        sc = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag)
        g = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = ELO_K * g * (sc - e)
        r[h] += delta
        r[a] -= delta
    return r, pre


def fit_euro(M, C, T, extra_teams=(), init=None):
    """Dixon-Coles over the five leagues and the Champions League together."""
    m = (M["date"] < T) & (M["date"] >= T - WINDOW_DAYS)
    c = (C["date"] < T) & (C["date"] >= T - WINDOW_DAYS)
    w = np.concatenate([np.exp(-XI * (T - M["date"][m])), CL_WEIGHT * np.exp(-XI * (T - C["date"][c]))])
    fit = fit_poisson(np.concatenate([M["home"][m], C["home"][c]]), np.concatenate([M["away"][m], C["away"][c]]),
                      np.concatenate([M["hg"][m], C["hg"][c]]), np.concatenate([M["ag"][m], C["ag"][c]]), w,
                      extra_teams=extra_teams, init=init, ridge_base=EURO_RIDGE,
                      hm=np.concatenate([np.ones(int(m.sum())), (~C["neutral"][c]).astype(float)]))
    seen = defaultdict(int)
    for t in np.concatenate([M["home"][m], M["away"][m], C["home"][c], C["away"][c]]):
        seen[t] += 1
    return fit, seen


def backtest_cl(M, C, elo_pre):
    cur = season_of(date.today())
    seasons = sorted(s for s in set(C["season"]) if s < cur)
    if len(seasons) < CL_BACKTEST_SEASONS + 3:
        return None
    bt = seasons[-CL_BACKTEST_SEASONS:]
    test = np.where(np.isin(C["season"], bt))[0]
    start = int(C["date"][test].min())
    train = (C["season"] >= seasons[2]) & (C["date"] < start)
    logit_c = fit_logit(elo_pre[train], C["res"][train])
    weeks = C["date"][test] - (C["date"][test] % 7)
    idx, dc_P, dc_O, dc_B = [], [], [], []
    prev = None
    for wk in np.unique(weeks):
        block = test[weeks == wk]
        fit, seen = fit_euro(M, C, int(wk), extra_teams=set(C["home"][block]) | set(C["away"][block]), init=prev)
        prev = fit
        for i in block:
            h, a = C["home"][i], C["away"][i]
            if seen[h] < 4 or seen[a] < 4:
                continue
            mk = grid_markets(score_grid(*fit.rates(h, a, bool(C["neutral"][i])), fit.rho))
            idx.append(i)
            dc_P.append([mk["H"], mk["D"], mk["A"]])
            dc_O.append(mk["O25"])
            dc_B.append(mk["BTTS"])
    J = np.array(idx)
    y = C["res"][J]
    dcP = np.array(dc_P)
    eloP = logit_probs(elo_pre[J], logit_c)
    ensP = (dcP + eloP) / 2
    out = {"seasons": bt, "matches": int(len(J)), "elo_logit": logit_c,
           "home_win_rate": float(np.mean(y == 0)), "models": {}}
    for name, P in (("Dixon-Coles", dcP), ("Elo", eloP), ("Ensemble", ensP)):
        out["models"][name] = metrics(P, y)
    over = (C["hg"][J] + C["ag"][J] > 2).astype(float)
    btts = ((C["hg"][J] > 0) & (C["ag"][J] > 0)).astype(float)
    out["over25"] = {"Dixon-Coles": binary_logloss(np.array(dc_O), over),
                     "Base rate": binary_logloss(np.full(len(over), over.mean()), over)}
    out["btts"] = {"Dixon-Coles": binary_logloss(np.array(dc_B), btts),
                   "Base rate": binary_logloss(np.full(len(btts), btts.mean()), btts)}
    pp, hit = ensP.ravel(), np.eye(3)[y].ravel()
    bins = np.clip((pp * 10).astype(int), 0, 9)
    out["calibration"] = [
        {"bin": f"{b * 10}-{b * 10 + 10}%", "predicted": float(pp[bins == b].mean()),
         "actual": float(hit[bins == b].mean()), "n": int((bins == b).sum())}
        for b in range(10) if (bins == b).sum() >= 30
    ]
    return out


def form_cl(C, team, k=5):
    idx = np.where((C["home"] == team) | (C["away"] == team))[0][-k:]
    out = []
    for i in idx:
        home = C["home"][i] == team
        gf, ga = (C["hg"][i], C["ag"][i]) if home else (C["ag"][i], C["hg"][i])
        out.append({"r": "W" if gf > ga else "D" if gf == ga else "L", "s": f"{gf}-{ga}",
                    "opp": C["away"][i] if home else C["home"][i],
                    "ha": "N" if C["neutral"][i] else "H" if home else "A",
                    "d": date.fromordinal(int(C["date"][i])).isoformat(), "c": "Champions League"})
    return out


def final_cl(M, C, ratings, logit_c, fixtures):
    T = date.today().toordinal() + 1
    recent = sorted(set(C["season"]))[-2:]
    teams_now = set(C["home"][np.isin(C["season"], recent)]) | set(C["away"][np.isin(C["season"], recent)])
    teams_now |= {f["home"] for f in fixtures if f["div"] == CL} | {f["away"] for f in fixtures if f["div"] == CL}
    fit, _ = fit_euro(M, C, T, extra_teams=teams_now)
    teams = {}
    for t in sorted(teams_now):
        i = fit.idx[t]
        teams[t] = {"att": round(float(fit.att[i]), 4), "def": round(float(fit.de[i]), 4),
                    "elo": round(ratings.get(t, CL_NEW_ELO), 1), "promoted": False, "form": form_cl(C, t)}
    log(f"  {CL_NAME}: {len(teams)} teams, home adv x{math.exp(fit.hfa):.2f}, rho {fit.rho:+.3f}")
    return {"name": CL_NAME, "european": True, "neutral_option": True, "mu0": round(float(fit.mu0), 4),
            "hfa": round(float(fit.hfa), 4), "rho": round(float(fit.rho), 4), "sot": None,
            "elo": {"hfa": ELO_HFA, "logit": logit_c}, "teams": teams}


# ---------------------------------------------------------------- final fit + export
def form_for(M, L, team, k=5):
    m = (M["div"] == L) & ((M["home"] == team) | (M["away"] == team))
    idx = np.where(m)[0][-k:]
    out = []
    for i in idx:
        home = M["home"][i] == team
        gf, ga = (M["hg"][i], M["ag"][i]) if home else (M["ag"][i], M["hg"][i])
        out.append({"r": "W" if gf > ga else "D" if gf == ga else "L",
                    "s": f"{gf}-{ga}", "opp": M["away"][i] if home else M["home"][i],
                    "ha": "H" if home else "A",
                    "d": date.fromordinal(int(M["date"][i])).isoformat()})
    return out


def final_models(M, elo, fixtures):
    today = date.today()
    T = today.toordinal() + 1
    season = season_of(today)
    leagues = {}
    for L, name in LEAGUES.items():
        inL = M["div"] == L
        if not inL.any():
            continue
        cur = set(M["home"][inL & (M["season"] == season)]) | set(M["away"][inL & (M["season"] == season)])
        cur |= {f["home"] for f in fixtures if f["div"] == L} | {f["away"] for f in fixtures if f["div"] == L}
        if not cur:  # pre-season with no fixtures yet: use the latest season's teams
            last = max(M["season"][inL])
            cur = set(M["home"][inL & (M["season"] == last)])
        promo = promoted_teams(M, L, season, cur)
        g = fit_window(M, L, T, promoted=promo, extra_teams=cur)
        s = fit_window(M, L, T, promoted=promo, extra_teams=cur, what="sot")
        stat_fits = {k: fit_stat(M, L, T, k, promoted=promo, extra_teams=cur) for k in STATS}
        teams = {}
        for t in sorted(cur):
            gi = g.idx[t]
            entry = {"att": round(float(g.att[gi]), 4), "def": round(float(g.de[gi]), 4),
                     "elo": round(elo.rating_for(L, season, t), 1),
                     "promoted": t in promo, "form": form_for(M, L, t)}
            if s is not None:
                si = s.idx[t]
                entry["sot_att"] = round(float(s.att[si]), 4)
                entry["sot_def"] = round(float(s.de[si]), 4)
            entry["stats"] = {k: [round(float(f.att[f.idx[t]]), 4), round(float(f.de[f.idx[t]]), 4)]
                              for k, (f, _) in stat_fits.items() if f is not None}
            teams[t] = entry
        leagues[L] = {
            "name": name, "mu0": round(float(g.mu0), 4), "hfa": round(float(g.hfa), 4),
            "rho": round(float(g.rho), 4),
            "sot": None if s is None else {"mu0": round(float(s.mu0), 4), "hfa": round(float(s.hfa), 4)},
            "stats": {k: {"label": STATS[k][2], "line": STATS[k][3], "mu0": round(float(f.mu0), 4),
                          "hfa": round(float(f.hfa), 4), "alpha": round(a, 4)}
                      for k, (f, a) in stat_fits.items() if f is not None},
            "teams": teams,
        }
        log(f"  {name}: {len(teams)} teams, home adv x{math.exp(g.hfa):.2f}, rho {g.rho:+.3f}"
            + (f", promoted: {', '.join(sorted(promo))}" if promo else ""))
    return leagues


def predict(leagues, logit_c, L, home, away, neutral=False):
    lg = leagues[L]
    e = lg.get("elo") or {"hfa": ELO_HFA, "logit": logit_c}
    th, ta = lg["teams"][home], lg["teams"][away]
    lam = math.exp(lg["mu0"] + (0.0 if neutral else lg["hfa"]) + th["att"] + ta["def"])
    mu = math.exp(lg["mu0"] + ta["att"] + th["def"])
    mk = grid_markets(score_grid(lam, mu, lg["rho"]))
    eloP = logit_probs(th["elo"] + (0.0 if neutral else e["hfa"]) - ta["elo"], e["logit"])
    dc = [mk["H"], mk["D"], mk["A"]]
    ens = [(a + b) / 2 for a, b in zip(dc, eloP)]
    return {"dc": dc, "elo": [float(v) for v in eloP], "ens": ens, "O25": mk["O25"], "BTTS": mk["BTTS"]}


# ---------------------------------------------------------------- prediction log
LOG_FIELDS = ["logged", "Date", "Div", "HomeTeam", "AwayTeam",
              "dc_H", "dc_D", "dc_A", "elo_H", "elo_D", "elo_A", "ens_H", "ens_D", "ens_A",
              "dc_O25", "dc_BTTS", "odds_H", "odds_D", "odds_A"]


def update_log(leagues, logit_c, fixtures):
    rows = {}
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows[(r["Div"], r["HomeTeam"], r["AwayTeam"], r["Date"])] = r
    today = date.today()
    horizon = date.fromordinal(today.toordinal() + LOG_DAYS_AHEAD).isoformat()
    for fx in fixtures:
        L = fx["div"]
        if fx["date"] > horizon:
            continue  # too far ahead; it gets logged closer to kick-off
        if L not in leagues or fx["home"] not in leagues[L]["teams"] or fx["away"] not in leagues[L]["teams"]:
            continue
        p = predict(leagues, logit_c, L, fx["home"], fx["away"], bool(fx.get("neutral")))
        r = {"logged": today.isoformat(), "Date": fx["date"], "Div": L,
             "HomeTeam": fx["home"], "AwayTeam": fx["away"]}
        for k, key in enumerate("HDA"):
            r[f"dc_{key}"] = f"{p['dc'][k]:.4f}"
            r[f"elo_{key}"] = f"{p['elo'][k]:.4f}"
            r[f"ens_{key}"] = f"{p['ens'][k]:.4f}"
            o = (fx.get("odds") or {}).get(key, math.nan)
            r[f"odds_{key}"] = "" if o is None or math.isnan(o) else f"{o:.2f}"
        r["dc_O25"], r["dc_BTTS"] = f"{p['O25']:.4f}", f"{p['BTTS']:.4f}"
        rows[(L, fx["home"], fx["away"], fx["date"])] = r  # latest pre-match prediction wins
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: (r["Date"], r["Div"], r["HomeTeam"])))
    return list(rows.values())


def results_index(M, I, C=None):
    """(competition, home, away) -> [(day, home goals, away goals)] for clubs and countries."""
    res = defaultdict(list)
    for i in range(M["n"]):
        res[(M["div"][i], M["home"][i], M["away"][i])].append((int(M["date"][i]), int(M["hg"][i]), int(M["ag"][i])))
    if I is not None:
        recent = np.where(I["date"] >= date.today().toordinal() - 3 * 365)[0]
        for i in recent:
            res[(INT, I["home"][i], I["away"][i])].append((int(I["date"][i]), int(I["hg"][i]), int(I["ag"][i])))
    if C is not None:
        for i in np.where(C["date"] >= date.today().toordinal() - 365)[0]:
            res[(CL, C["home"][i], C["away"][i])].append((int(C["date"][i]), int(C["hg"][i]), int(C["ag"][i])))
    return res


def track_record(results, log_rows):
    """Score logged predictions against results that have since come in."""
    settled = []
    for r in log_rows:
        d = date.fromisoformat(r["Date"]).toordinal()
        for day, hg, ag in results.get((r["Div"], r["HomeTeam"], r["AwayTeam"]), []):
            if abs(day - d) <= 30 and r["logged"] <= date.fromordinal(day).isoformat():
                settled.append((r, hg, ag))
                break
    if not settled:
        return {"settled": 0, "models": {}, "recent": []}
    y = np.array([0 if hg > ag else 1 if hg == ag else 2 for _, hg, ag in settled])
    out = {"settled": len(settled), "models": {}, "recent": []}
    for name, key in (("Dixon-Coles", "dc"), ("Elo", "elo"), ("Ensemble", "ens")):
        P = np.array([[float(r[f"{key}_{k}"]) for k in "HDA"] for r, _, _ in settled])
        out["models"][name] = metrics(P, y)
    bo = np.array([[float(r[f"odds_{k}"]) if r[f"odds_{k}"] else math.nan for k in "HDA"] for r, _, _ in settled])
    has = np.all(np.isfinite(bo), axis=1)
    if has.any():
        bp = (1 / bo[has]) / (1 / bo[has]).sum(axis=1, keepdims=True)
        out["models"]["Bookmakers"] = metrics(bp, y[has])
    for r, hg, ag in sorted(settled, key=lambda t: t[0]["Date"], reverse=True)[:40]:
        out["recent"].append({
            "date": r["Date"], "div": r["Div"], "home": r["HomeTeam"], "away": r["AwayTeam"],
            "score": f"{hg}-{ag}", "res": "H" if hg > ag else "D" if hg == ag else "A",
            "ens": [float(r[f"ens_{k}"]) for k in "HDA"],
        })
    return out


def clean(o):
    """Make floats JSON-safe and compact."""
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else round(o, 4)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return clean(o.item())
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-backtest", action="store_true", help="reuse the last backtest results")
    args = ap.parse_args()

    t0 = time.time()
    M = load_matches()
    last = date.fromordinal(int(M["date"].max()))
    log(f"Loaded {M['n']} club matches, latest result {last}")
    fixtures = load_fixtures(M)
    log(f"Upcoming club fixtures: {len(fixtures)}")
    I = load_international()
    if I is not None:
        log(f"Loaded {I['n']} international matches, latest {date.fromordinal(int(I['date'].max()))}")

    C = load_cl()
    if C is not None:
        log(f"Loaded {C['n']} Champions League matches, latest {date.fromordinal(int(C['date'].max()))}")

    log("Running Elo through every match...")
    elo, elo_pre = run_elo(M)
    if I is not None:
        int_ratings, int_pre = run_elo_int(I)
    if C is not None:
        euro_ratings, euro_pre = run_elo_euro(M, C)

    bt = bt_int = bt_cl = None
    if args.skip_backtest and BACKTEST_FILE.exists():
        bt = json.loads(BACKTEST_FILE.read_text(encoding="utf-8"))
        log("Using saved club backtest results")
    else:
        log(f"Walk-forward backtest over the last {BACKTEST_SEASONS} complete seasons...")
        bt = backtest(M, elo_pre)
        if bt is not None:
            log("Match statistics backtest (shots, corners, cards)...")
            bt["match_stats"] = backtest_stats(M)
            BACKTEST_FILE.write_text(json.dumps(clean(bt), indent=1), encoding="utf-8")
    if I is not None:
        if args.skip_backtest and BACKTEST_INT_FILE.exists():
            bt_int = json.loads(BACKTEST_INT_FILE.read_text(encoding="utf-8"))
        else:
            log(f"National teams backtest over the last {INT_BACKTEST_YEARS} years...")
            bt_int = backtest_int(I, int_pre)
            BACKTEST_INT_FILE.write_text(json.dumps(clean(bt_int), indent=1), encoding="utf-8")

    if C is not None:
        if args.skip_backtest and BACKTEST_CL_FILE.exists():
            bt_cl = json.loads(BACKTEST_CL_FILE.read_text(encoding="utf-8"))
        else:
            log(f"Champions League backtest over the last {CL_BACKTEST_SEASONS} seasons...")
            bt_cl = backtest_cl(M, C, euro_pre)
            if bt_cl is not None:
                BACKTEST_CL_FILE.write_text(json.dumps(clean(bt_cl), indent=1), encoding="utf-8")

    if bt is not None:
        logit_c = bt["elo_logit"]
    else:
        recent = M["date"] >= M["date"].max() - 6 * 365
        logit_c = fit_logit(elo_pre[recent], M["res"][recent])

    log("Fitting current models...")
    leagues = final_models(M, elo, fixtures)
    if I is not None:
        leagues[INT] = final_int(I, int_ratings, bt_int["elo_logit"])
        for u in I["upcoming"]:  # the source lists upcoming internationals only occasionally
            fixtures.append({"div": INT, "date": u["date"], "time": "", "tz": "",
                             "home": u["home_team"], "away": u["away_team"], "odds": None, "best": None,
                             "neutral": (u.get("neutral") or "").strip().upper() == "TRUE",
                             "comp": u.get("tournament") or ""})
        fixtures.sort(key=lambda f: (f["date"], f["time"], f["div"]))
    if C is not None:
        cl_logit = bt_cl["elo_logit"] if bt_cl else fit_logit(euro_pre, C["res"])
        leagues[CL] = final_cl(M, C, euro_ratings, cl_logit, fixtures)
    log_rows = update_log(leagues, logit_c, fixtures)
    track = track_record(results_index(M, I, C), log_rows)
    for wmsg in WARNINGS:
        log("  ! " + wmsg)

    payload = clean({
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_through": last.isoformat(),
        "int_through": None if I is None else date.fromordinal(int(I["date"].max())).isoformat(),
        "matches": M["n"],
        "int_matches": 0 if I is None else I["n"],
        "elo": {"hfa": ELO_HFA, "logit": logit_c},
        "leagues": leagues,
        "fixtures": [f for f in fixtures if f["div"] in leagues
                     and f["home"] in leagues[f["div"]]["teams"] and f["away"] in leagues[f["div"]]["teams"]],
        "backtest": bt,
        "backtest_int": bt_int,
        "backtest_cl": bt_cl,
        "cl_through": None if C is None else date.fromordinal(int(C["date"].max())).isoformat(),
        "cl_matches": 0 if C is None else C["n"],
        "warnings": WARNINGS,
        "sheet": SHEET,
        "track": track,
        "settings": {"xi": XI, "window_days": WINDOW_DAYS, "elo_k": ELO_K, "value_edge": VALUE_EDGE},
    })
    APP_DATA.parent.mkdir(parents=True, exist_ok=True)
    APP_DATA.write_text("window.MODEL_DATA = " + json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + ";\n",
                        encoding="utf-8")
    log(f"Exported {APP_DATA} in {time.time() - t0:.0f}s")
    if bt is not None:
        log("\nClub backtest (home/draw/away log loss, lower is better):")
        for k, v in bt["models"].items():
            if v.get("logloss") is not None:
                log(f"  {k:12s} {v['logloss']:.4f}   accuracy {v['accuracy'] * 100:.1f}%")
    if bt_cl is not None:
        log("Champions League backtest:")
        for k, v in bt_cl["models"].items():
            log(f"  {k:12s} {v['logloss']:.4f}   accuracy {v['accuracy'] * 100:.1f}%")
    if bt_int is not None:
        log("National teams backtest:")
        for k, v in bt_int["models"].items():
            log(f"  {k:12s} {v['logloss']:.4f}   accuracy {v['accuracy'] * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
