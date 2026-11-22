@echo off
rem Nightly daily-loop run (Windows Task Scheduler, weekdays 18:45 and 21:00).
rem Safe to run twice: predictions are written once per day, settled days are skipped.
cd /d "%~dp0..\.."
".venv\Scripts\python.exe" -m code.research.daily >> "code\data\history\daily.log" 2>&1
