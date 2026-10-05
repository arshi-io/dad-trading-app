@echo off
rem Starts Papa Terminal for phones on the same Wi-Fi. Settings and secrets come from .env.
cd /d "%~dp0"
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address" ^| findstr /v "10.5."') do set LANIP=%%a
echo.
echo   Papa Terminal is starting. On the phone (same Wi-Fi) open:  http://%LANIP: =%:8731
echo   Keep this window open. Close it to stop the server.
echo.
".venv\Scripts\python.exe" -m uvicorn code.app.main:app --host 0.0.0.0 --port 8731 --no-server-header
pause
