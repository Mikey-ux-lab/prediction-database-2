# Match Odds Lab

A probability calculator for football. It covers:

- the Premier League, La Liga, Bundesliga, Serie A and Ligue 1 (every result since 2006/07)
- the Champions League (every match since 2011/12)
- national teams (friendlies, World Cups, Nations Leagues, AFCON, Copa América, Euros and their qualifiers)

It retrains itself each day as new results come in.

## Try it straight away

Open `app\index.html` in Chrome or Edge. The folder already contains the data and a trained model, so the calculator works before you install anything.

## Turn on updates (Windows)

1. Install Python 3.10 or newer from python.org. On the first installer screen, tick **Add python.exe to PATH**.
2. Put this whole folder somewhere permanent, for example `Documents\match-odds-lab`.
3. Double-click **setup.bat**. It installs numpy and scipy, fetches the latest results and fixtures, retrains the models and opens the calculator.
4. Double-click **schedule_daily.bat** so it updates every day at 18:00. If the laptop is off at 18:00, it runs the next time the laptop is on.

To update by hand at any time, double-click **update.bat**.

## Hosting it on GitHub (free, updates itself)

GitHub can run the update twice a day on its own servers and publish the page, so no laptop is needed.

1. Create a **public** repository on github.com and upload every file and folder from this project, including the `.github` folder. Do not upload `api_key.txt`.
2. In the repository, open **Settings → Pages** and set **Source** to **GitHub Actions**.
3. Open the **Actions** tab, choose **Update predictions**, and click **Run workflow**. After a few minutes the page is live at `https://<your-username>.github.io/<repository-name>/`.
4. To use a football-data.org key, open **Settings → Secrets and variables → Actions → New repository secret**, name it `FOOTBALL_DATA_KEY`, and paste the key as the value. Never put a key in a file in a public repository.
5. To add national-team results, open `data/my_matches.csv` on GitHub, click the pencil, add rows and commit. That starts an update on its own.

The workflow is in `.github/workflows/update.yml`. It runs at 05:17 and 17:17 GMT; GitHub sometimes starts scheduled runs late.

## Daily league and Champions League data (free key)

Without a key, league results come from a file that refreshes about weekly, and there are no Champions League fixtures. With a free football-data.org key you get results and fixtures for the five leagues and the Champions League on every update.

1. Register at football-data.org/client/register. The key arrives by email.
2. Open `api_key.txt` in Notepad, paste the key on the empty line under the two `#` lines, and save.
3. Double-click **update.bat**. The output should say "Using your football-data.org key".

Keep the key private. Kick-off times from this source are shown in GMT, which is Ghana time.

## National teams: your spreadsheet

No free service gives daily national-team results, so you add them to `data\my_matches.csv`. Open it in Excel, add one row per match, save it (keep the CSV format), then run **update.bat**.

| Column | What to type |
|---|---|
| date | `2026-10-03` or `03/10/2026` |
| home_team, away_team | Country names as the model spells them. The full list is in `data\country_names.txt`. |
| home_score, away_score | The 90-minute score. **Leave both blank for an upcoming match** and it appears in the fixture list. |
| tournament | For example `UEFA Nations League`, `African Cup of Nations qualification`, `FIFA World Cup qualification`, `Friendly` |
| neutral | `yes` if neither team is at home, otherwise leave blank |

`data\my_matches_example.csv` shows three example rows. If a name is misspelt, or an upcoming match is dated in the past, the calculator shows a yellow notice at the top. A line under "Any two teams" says how many of your rows are in use, so you can confirm an update picked them up. When the public dataset catches up with a match you entered, your row is ignored automatically, so there is no double counting.

## Using the calculator

- **Date strip**: pick a day to see its matches, grouped by competition. The three numbers on each row are the home, draw and away chances in percent.
- **Any two teams**: predict any pairing. For the Champions League and national teams, tick **Neutral venue** for finals and tournament games in a third country.
- **Team news adjustment**: if key players are out, drag a team's attack or defence down. Every probability updates. This is your judgement; the model doesn't read line-ups.
- **Markets**: 1X2, double chance, over/under 0.5–5.5, both teams to score, team goals, multigoals and combos. Type your SportyBet odds into **Your odds**. **Edge** is the expected profit per 1 staked if the model is right.
- **Correct score**: tap cells to build a multiscore and see its combined chance and fair odds.
- **Match stats**: expected shots, shots on target, corners, yellow cards, red cards and fouls for each team, with over/under lines you can price against your bookmaker. League matches only. These numbers refresh when football-data.co.uk posts new results (about twice a week), because the other sources carry scores only.
- **Goalkeeper saves**: an estimate built from shots on target, for league matches only.
- **Team ratings**: Elo ranking plus expected goals for and against.
- **How good is it?**: backtests for leagues, Champions League and national teams, calibration charts, and the live record of past predictions.

## How it learns

| Model | What updates after each result |
|---|---|
| Elo | Both teams' ratings move straight away. Bigger wins move them more. For national teams, a World Cup match moves ratings three times as much as a friendly. |
| Dixon-Coles | Attack and defence strengths are refitted on every run. For clubs, a match from a year ago counts about half as much as yesterday's. For countries the fade is slower, and a friendly counts 60% of a competitive match. |
| Ensemble | The average of the two for home/draw/away. It is the headline number. |

For the Champions League, clubs are rated on one scale fitted to the five leagues and European matches together. Clubs from outside the five leagues are rated only on their European matches, so their numbers are less certain.

Every run saves predictions for fixtures in the next 10 days in `data\prediction_log.csv`. When the results arrive, the **Live record** scores them.

## Where the data comes from

| Data | Source | Refreshed |
|---|---|---|
| League results, shots, bookmaker odds | football-data.co.uk | About twice a week |
| Next round of league fixtures with odds | football-data.co.uk | Fridays and Tuesdays |
| League and Champions League results and fixtures | football-data.org (needs your free key) | Every update |
| League schedule and scores, when there is no key | `github.com/openfootball/football.json` | Weekly |
| National team results | `github.com/martj42/international_results`, plus your spreadsheet | After each international window |
| Bundled league history | `github.com/datasets/football-datasets`, `github.com/xgabora/Club-Football-Match-Data-2000-2025` | One-off |
| Bundled Champions League history | `github.com/openfootball/champions-league` | One-off |

If a source is unreachable, the update carries on with the others and with the files already in `data\`.

## Settings

The constants at the top of `model.py` control the models:

- `XI`, `INT_XI`: how fast old matches fade for clubs and countries. Higher reacts faster but is noisier.
- `ELO_K`: how much one result moves a club's Elo rating.
- `PROMOTED_PRIOR`: the starting strength for newly promoted teams.
- `INT_FRIENDLY_WEIGHT`: how much a friendly counts next to a competitive match.
- `CL_WEIGHT`, `EURO_RIDGE`: how much European matches count, and how firmly thinly-observed clubs are pulled toward average.
- `VALUE_EDGE`: the edge threshold used in the backtest.

After changing one, run `python model.py` and compare the backtest log loss. Lower is better.

## Files

| File | Purpose |
|---|---|
| `api_key.txt` | Your football-data.org key (optional) |
| `data\my_matches.csv` | National-team results and fixtures you add |
| `data\country_names.txt` | The country spellings the model accepts |
| `data\history.csv` | Every league match since 2006/07, bundled |
| `data\champions_league.csv` | Champions League matches 2011/12 to 2025/26, bundled |
| `data\international.csv` | Every international, replaced on each update |
| `data\schedule\` | This season's fixture lists |
| `data\team_names.json` | Converts source names ("Manchester City FC") to the model's ("Man City"). Edit it if a new team is matched wrongly. |
| `download_data.py` | Fetches results, schedules and fixtures, and merges them with the history |
| `model.py` | Trains, backtests, logs predictions, writes `app\model_data.js` |
| `app\index.html` | The calculator (opens offline; fonts load when online) |
| `data\update_log.txt` | Output of scheduled runs. Check here if something looks stale. |

## Honest limits

- The models only know past scores (and shots, for leagues). They don't see injuries, line-ups, rotation, weather or motivation, and bookmakers do.
- In the three-season league backtest the bookmakers' probabilities were sharper, and betting every 5%+ "edge" lost about 10–13% of stakes.
- In the Champions League backtest the both-teams-to-score numbers were no better than quoting the average rate.
- Match stats: shots, shots on target, yellow cards and fouls beat the league average in testing; corners only barely; red cards not at all. Referees, which drive card counts, aren't in the data.
- There are no possession, passing, injury, player or line-up numbers. No free public source keeps those current for all five leagues.
- Friendlies are the least predictable internationals because coaches experiment.
- Most betting systems lose money in the long run, so only stake what you can afford to lose.
