@echo off
REM One-time setup: installs Python packages, fetches the latest results and
REM fixtures, retrains the models and opens the calculator.
REM The 20 years of history is already in data\history.csv.
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")

echo.
echo === 1/3  Installing Python packages (numpy, scipy) ===
%PY% -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
  echo.
  echo Could not install packages. Check that Python is installed and on PATH:
  echo   https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^)
  pause
  exit /b 1
)

echo.
echo === 2/3  Fetching this season's results and upcoming fixtures ===
%PY% download_data.py --current
if errorlevel 1 ( echo Download step failed. & pause & exit /b 1 )

echo.
echo === 3/3  Training the models and running the backtest ===
%PY% model.py
if errorlevel 1 ( echo Model training failed. & pause & exit /b 1 )

echo.
echo Done. Opening the calculator...
start "" "%~dp0app\index.html"
echo.
echo Next: double-click schedule_daily.bat to update automatically every day.
pause
