#!/usr/bin/env python3
"""
Download match data for Europe's top 5 leagues from football-data.co.uk.

Usage:
    python download_data.py --current    # current season + upcoming fixtures (daily)
    python download_data.py --all        # optional: re-download every season since 2006/07

The project ships with data/history.csv (every match from 2006/07, built from
public GitHub mirrors of football-data.co.uk), so --current is all you need.
Any season file downloaded into data/raw/ replaces that season in the history.

Output:
    data/raw/<season>_<league>.csv   one file per downloaded season and league
    data/matches.csv                 history + downloads merged, sorted by date
    data/fixtures.csv                upcoming top-5 fixtures with bookmaker odds

Standard library only.
"""
import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}
FIRST_START_YEAR = 2006  # 2006/07

# Full-season schedules (every fixture to the end of the season) and national
# team results come from public GitHub datasets.
SCHEDULE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"
SCHEDULE_CODES = {"E0": "en.1", "SP1": "es.1", "D1": "de.1", "I1": "it.1", "F1": "fr.1"}
INTL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Optional: a free football-data.org key in api_key.txt gives daily results and
# fixtures for the five leagues and the Champions League.
API_URL = "https://api.football-data.org/v4/competitions/{code}/matches?season={year}"
API_CODES = {"E0": "PL", "SP1": "PD", "D1": "BL1", "I1": "SA", "F1": "FL1", "CL": "CL"}
CL = "CL"
STAGES = {"LEAGUE_STAGE": "League phase", "GROUP_STAGE": "Group stage", "PLAYOFFS": "Knockout play-offs",
          "LAST_16": "Round of 16", "QUARTER_FINALS": "Quarter-finals", "SEMI_FINALS": "Semi-finals",
          "FINAL": "Final", "REGULAR_SEASON": ""}

# Columns kept. Older seasons lack some; they stay blank.
KEEP = [
    "Div", "Season", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5",
    "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5",
    "Max>2.5", "Max<2.5",
    # Older seasons (to 2018/19) used BetBrain column names for max/average odds
    "BbMxH", "BbMxD", "BbMxA", "BbAvH", "BbAvD", "BbAvA",
    "BbMx>2.5", "BbMx<2.5", "BbAv>2.5", "BbAv<2.5",
    "PSH", "PSD", "PSA",  # Pinnacle
]
FIXTURE_KEEP = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5",
    "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5",
    "Max>2.5", "Max<2.5", "PSH", "PSD", "PSA",
]

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
HISTORY_FILE = HERE / "data" / "history.csv"
OUT_FILE = HERE / "data" / "matches.csv"
FIX_FILE = HERE / "data" / "fixtures.csv"
FD_FIX_FILE = HERE / "data" / "fixtures_footballdata.csv"
SCHEDULE_DIR = HERE / "data" / "schedule"
NAMES_FILE = HERE / "data" / "team_names.json"
INTL_FILE = HERE / "data" / "international.csv"
KEY_FILE = HERE / "api_key.txt"


def current_start_year(today=None):
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_code(start_year):
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def fetch_url(url, label, retries=3, starts_with="div"):
    """Return decoded text, or None if unavailable or not the expected file."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")  # older seasons use Latin-1
            # Guard against getting an HTML page instead of data.
            head = text.lstrip()[:300].lower()
            if "<html" in head or not head.startswith(starts_with):
                print(f"  ! {label}: not the expected file (got HTML or empty), skipped")
                return None
            return text
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  - {label}: not available yet")
                return None
            print(f"  ! {label}: HTTP {e.code}, retry {attempt + 1}")
        except Exception as e:  # network hiccup
            print(f"  ! {label}: {e}, retry {attempt + 1}")
        time.sleep(2)
    return None


def parse_date(s):
    s = (s or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def download(start_years):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    got = 0
    for sy in start_years:
        code = season_code(sy)
        print(f"Season {sy}/{sy + 1}")
        for league in LEAGUES:
            text = fetch_url(BASE.format(season=code, league=league), f"{code} {league}")
            if text is None:
                continue
            (RAW_DIR / f"{code}_{league}.csv").write_text(text, encoding="utf-8")
            got += 1
            time.sleep(0.5)  # be polite to the server
    print(f"Downloaded {got} files")
    return got


def merge():
    rows = []
    covered = set()  # (season, league) pairs we have a fresh download for
    for path in sorted(RAW_DIR.glob("*.csv")):
        code = path.name.split("_")[0]
        season = f"20{code[:2]}/{code[2:]}"
        fresh = 0
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("HomeTeam") or not r.get("AwayTeam"):
                    continue
                if (r.get("FTHG") or "").strip() == "" or (r.get("FTAG") or "").strip() == "":
                    continue  # unplayed or incomplete match
                d = parse_date(r.get("Date"))
                if d is None:
                    continue
                r["Season"] = season
                r["Date"] = d.isoformat()
                rows.append({k: (r.get(k) or "").strip() for k in KEEP})
                fresh += 1
        if fresh:
            covered.add((season, path.stem.split("_", 1)[1]))
    kept = 0
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if (r["Season"], r["Div"]) not in covered:
                    rows.append({k: (r.get(k) or "").strip() for k in KEEP})
                    kept += 1
    fresh_n = len(rows) - kept
    # Results that football-data hasn't posted yet: take the score from the schedule.
    have = {(r["Season"], r["Div"], r["HomeTeam"], r["AwayTeam"]) for r in rows}
    extra = 0
    for m in load_schedule():
        if m["score"] is None or m["div"] not in LEAGUES:
            continue
        key = (m["season"], m["div"], m["home"], m["away"])
        if key in have:
            continue
        hg, ag = m["score"]
        row = {k: "" for k in KEEP}
        row.update(Div=m["div"], Season=m["season"], Date=m["date"], HomeTeam=m["home"], AwayTeam=m["away"],
                   FTHG=str(hg), FTAG=str(ag), FTR="H" if hg > ag else "D" if hg == ag else "A")
        rows.append(row)
        have.add(key)
        extra += 1
    print(f"Bundled history: {kept} matches, fresh downloads: {fresh_n}, scores from schedule: {extra}")
    rows.sort(key=lambda r: (r["Date"], r["Div"], r["HomeTeam"]))
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEEP)
        w.writeheader()
        w.writerows(rows)
    print(f"Merged {len(rows)} played matches -> {OUT_FILE}")
    if rows:
        print(f"Date range: {rows[0]['Date']} to {rows[-1]['Date']}")


def download_fixtures():
    """Next round of fixtures with bookmaker odds, from football-data.co.uk."""
    text = fetch_url(FIXTURES_URL, "fixtures")
    if text is None:
        print("No football-data fixtures file; keeping the previous one if any")
        return
    FD_FIX_FILE.parent.mkdir(parents=True, exist_ok=True)
    FD_FIX_FILE.write_text(text, encoding="utf-8")


def read_api_key():
    env = os.environ.get("FOOTBALL_DATA_KEY", "").strip()   # set as a secret when hosted on GitHub
    if env:
        return env
    if not KEY_FILE.exists():
        return None
    for line in KEY_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def api_matches(code, year, key):
    """All matches of one competition and season from football-data.org, or None."""
    req = urllib.request.Request(API_URL.format(code=code, year=year),
                                 headers={"X-Auth-Token": key, "User-Agent": "match-odds-lab"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        why = {400: "the key was not accepted", 401: "the key was not accepted",
               403: "this competition is not in your plan, or the key is wrong",
               429: "too many requests, try again in a minute"}.get(e.code, f"HTTP {e.code}")
        print(f"  ! football-data.org {code}: {why}")
        return None
    except Exception as e:
        print(f"  ! football-data.org {code}: {e}")
        return None
    return data.get("matches") or None


def convert_api(matches):
    """football-data.org matches -> the schedule format used in data/schedule/."""
    out = []
    for m in matches:
        home = (m.get("homeTeam") or {}).get("name")
        away = (m.get("awayTeam") or {}).get("name")
        utc = m.get("utcDate") or ""
        status = m.get("status") or ""
        if not home or not away or len(utc) < 16:
            continue  # knockout tie whose teams are not known yet
        if status in ("POSTPONED", "CANCELLED", "SUSPENDED", "AWARDED"):
            continue
        row = {"stage": m.get("stage") or "", "date": utc[:10], "time": utc[11:16],
               "team1": home, "team2": away}
        if status == "FINISHED":
            sc = m.get("score") or {}
            ft = sc.get("regularTime") or sc.get("fullTime") or {}  # 90-minute score
            if ft.get("home") is not None and ft.get("away") is not None:
                row["score"] = {"ft": [int(ft["home"]), int(ft["away"])]}
        out.append(row)
    return out


def download_schedule(start_year):
    """Whole-season schedule and scores for each league and the Champions League."""
    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    season = f"{start_year}-{(start_year + 1) % 100:02d}"
    key = read_api_key()
    print("Using your football-data.org key" if key else
          "No key in api_key.txt: using the weekly GitHub schedule (no Champions League fixtures)")
    live = weekly = 0
    for div in list(SCHEDULE_CODES) + [CL]:
        data = None
        if key:
            ms = api_matches(API_CODES[div], start_year, key)
            if ms:
                data = {"matches": convert_api(ms), "_tz": "UTC", "_source": "football-data.org"}
                live += 1
            time.sleep(1)
        if data is None and div in SCHEDULE_CODES:
            text = fetch_url(SCHEDULE_URL.format(season=season, code=SCHEDULE_CODES[div]),
                             f"schedule {div}", starts_with="{")
            if text is not None:
                try:
                    data = json.loads(text)
                    data["_tz"], data["_source"] = "local", "openfootball"
                    weekly += 1
                except ValueError:
                    print(f"  ! schedule {div}: could not read the file, kept the previous one")
        if data is None:
            continue
        data["_season"] = f"{start_year}/{(start_year + 1) % 100:02d}"
        (SCHEDULE_DIR / f"{div}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"Schedules: {live} live from football-data.org, {weekly} from the weekly GitHub file")


def download_international():
    text = fetch_url(INTL_URL, "international results", starts_with="date,")
    if text is None:
        print("No international results download; keeping the bundled file")
        return
    INTL_FILE.parent.mkdir(parents=True, exist_ok=True)
    INTL_FILE.write_text(text, encoding="utf-8")
    print(f"Saved {max(text.count(chr(10)) - 1, 0)} international results -> {INTL_FILE}")


_STRIP = re.compile(r"\b(FC|AFC|CF|SC|AC|AS|SS|SSC|US|UD|CA|RC|RCD|OGC|AJ|ES|SV|VfB|VfL|TSG|BC|CFC|OSC|SCO|1\.|\d{2,4})\b", re.I)


def _known_names():
    names = {}
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                names.setdefault(r["Div"], set()).add(r["HomeTeam"])
    return names


def load_schedule():
    """All schedule matches with team names converted to the model's names."""
    if not SCHEDULE_DIR.exists():
        return []
    mapping = json.loads(NAMES_FILE.read_text(encoding="utf-8")) if NAMES_FILE.exists() else {}
    known = None
    added = 0
    out = []
    for div in list(SCHEDULE_CODES) + [CL]:
        path = SCHEDULE_DIR / f"{div}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        season = data.get("_season") or ""
        tz = data.get("_tz") or "local"
        table = mapping.setdefault(div, {})

        def name(n):
            nonlocal known
            if n in table:
                return table[n]
            if known is None:
                known = _known_names()
            pool = set(known.get(div, ()))
            if div == CL:  # any club can turn up in Europe
                pool = set(table.values()).union(*known.values()) if known else set(table.values())
            short = " ".join(_STRIP.sub(" ", n).split())
            close = difflib.get_close_matches(short, sorted(pool), n=1, cutoff=0.75)
            table[n] = close[0] if close else short
            nonlocal added
            added += 1
            print(f"  note: new team name '{n}' -> '{table[n]}' (edit data/team_names.json if wrong)")
            return table[n]

        for m in data.get("matches", []):
            if not m.get("date") or not m.get("team1") or not m.get("team2"):
                continue
            sc = m.get("score")
            # the GitHub source writes 0-0 results as a bare list instead of {"ft": [..]}
            ft = sc.get("ft") if isinstance(sc, dict) else sc if isinstance(sc, list) else None
            stage = m.get("stage") or ""
            out.append({"div": div, "season": season, "date": m["date"], "time": m.get("time") or "",
                        "tz": tz, "home": name(m["team1"]), "away": name(m["team2"]),
                        "stage": STAGES.get(stage, stage.replace("_", " ").capitalize()),
                        "neutral": div == CL and stage == "FINAL",
                        "score": tuple(ft) if ft and len(ft) == 2 else None})
    if added:  # remember new names so the note appears once and you can correct them
        NAMES_FILE.write_text(json.dumps(mapping, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    return out


def build_fixtures():
    """Upcoming fixtures: football-data's next round (with odds) + the full schedule."""
    today = date.today()
    cols = FIXTURE_KEEP + ["TZ", "Comp", "Neutral"]
    rows, seen = [], set()
    if FD_FIX_FILE.exists():
        with open(FD_FIX_FILE, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("Div") or "").strip() not in LEAGUES:
                    continue
                d = parse_date(r.get("Date"))
                if d is None or d < today:
                    continue
                r["Date"] = d.isoformat()
                row = {k: (r.get(k) or "").strip() for k in FIXTURE_KEEP}
                row.update(TZ="UK", Comp="", Neutral="")
                rows.append(row)
                seen.add((row["Div"], row["HomeTeam"], row["AwayTeam"]))
    with_odds = len(rows)
    for m in load_schedule():
        if m["score"] is not None or m["date"] < today.isoformat():
            continue
        if m["div"] != CL and (m["div"], m["home"], m["away"]) in seen:
            continue
        row = {k: "" for k in cols}
        row.update(Div=m["div"], Date=m["date"], Time=m["time"], HomeTeam=m["home"], AwayTeam=m["away"],
                   TZ=m["tz"], Comp=m["stage"], Neutral="TRUE" if m["neutral"] else "")
        rows.append(row)
    rows.sort(key=lambda r: (r["Date"], r["Time"], r["Div"]))
    FIX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FIX_FILE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    last = rows[-1]["Date"] if rows else "-"
    print(f"Saved {len(rows)} upcoming fixtures ({with_odds} with odds), through {last} -> {FIX_FILE}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="all seasons since 2006/07")
    g.add_argument("--current", action="store_true", help="current season only")
    args = ap.parse_args()

    cur = current_start_year()
    years = range(FIRST_START_YEAR, cur + 1) if args.all else [cur]
    download(years)
    download_schedule(cur)
    download_fixtures()
    download_international()
    merge()
    build_fixtures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
