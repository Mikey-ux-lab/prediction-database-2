@echo off
REM Creates a Windows scheduled task that runs update.bat every day at 18:00.
REM If the laptop is off at 18:00, it runs as soon as the laptop is next on.
cd /d "%~dp0"
set "TASK=MatchOddsLab update"
set "SCRIPT=%~dp0update.bat"
set "DIR=%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$q=[char]34; $a=New-ScheduledTaskAction -Execute ($q+$env:SCRIPT+$q) -Argument '/quiet' -WorkingDirectory $env:DIR; $t=New-ScheduledTaskTrigger -Daily -At '18:00'; $o=New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries; Register-ScheduledTask -TaskName $env:TASK -Action $a -Trigger $t -Settings $o -Force | Out-Null"

if errorlevel 1 (
  echo PowerShell method failed, trying schtasks instead...
  schtasks /Create /SC DAILY /ST 18:00 /TN "%TASK%" /TR "\"%SCRIPT%\" /quiet" /F
)
if errorlevel 1 (
  echo.
  echo Could not create the task. Right-click this file and choose "Run as administrator".
  pause
  exit /b 1
)

echo.
echo Scheduled "%TASK%" for 18:00 every day.
echo Each run writes its output to data\update_log.txt.
echo To remove it later, double-click remove_schedule.bat.
pause
