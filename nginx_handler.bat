@echo off
@REM start NGINX.exe

:check_process
timeout /t 6 /nobreak >nul 2>&1
tasklist /NH /FI "IMAGENAME eq SkyScope.exe" 2>nul | find /I /N "SkyScope.exe"
if not "%ERRORLEVEL%"=="1" goto check_process

timeout /t 1 /nobreak >nul 2>&1
nginx.exe -s stop

wait