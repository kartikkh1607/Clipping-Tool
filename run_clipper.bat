@echo off
title Clipper
cd /d "%~dp0"
call .venv\Scripts\activate.bat
pip install -q -U yt-dlp[default] >nul 2>nul
:loop
echo.
set "URL="
set /p URL=Paste YouTube link (or press Enter to quit):
if "%URL%"=="" exit /b 0
set "CLIPLANG=en"
set /p CLIPLANG=Language (en/hi) [en]:
if /i not "%CLIPLANG%"=="hi" set "CLIPLANG=en"
python clipper.py "%URL%" --lang %CLIPLANG%
goto loop
