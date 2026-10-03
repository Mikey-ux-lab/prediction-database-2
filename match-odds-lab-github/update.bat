@echo off
REM Daily update: fetch new results and fixtures, retrain, refresh the page data.
REM Double-click to run it yourself, or let the scheduled task run it with /quiet.
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
if not exist data mkdir data

if /i "%~1"=="/quiet" goto quiet

%PY% download_data.py --current
%PY% model.py
start "" "%~dp0app\index.html"
pause
exit /b 0

:quiet
echo. >> data\update_log.txt
echo ===== %date% %time% ===== >> data\update_log.txt
%PY% download_data.py --current >> data\update_log.txt 2>&1
%PY% model.py >> data\update_log.txt 2>&1
exit /b 0
