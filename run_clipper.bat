@echo off
title Clipper
cd /d "%~dp0"
call .venv\Scripts\activate.bat
pip install -q -U yt-dlp[default] >nul 2>nul
:loop
echo.
set "URL="
set /p URL=Paste YouTube or Twitch link (or press Enter to quit): 
if "%URL%"=="" exit /b 0
set "CAMP=none"
set /p CAMP=Campaign (skiimasktimmy / none) [none]: 
set "ARGS="
if /i "%CAMP%"=="none" goto asklang
set "ARGS=--campaign %CAMP%"
goto asksection
:asklang
set "CLIPLANG=en"
set /p CLIPLANG=Language (en/hi) [en]: 
if /i not "%CLIPLANG%"=="hi" set "CLIPLANG=en"
set "ARGS=--lang %CLIPLANG%"
:asksection
echo "%URL%" | find /i "twitch.tv/videos" >nul || goto run
set "SECTION="
set /p SECTION=Section (e.g. 0:30:00-2:00:00, Enter = whole VOD): 
if not "%SECTION%"=="" set "ARGS=%ARGS% --section %SECTION%"
:run
python clipper.py "%URL%" %ARGS%
goto loop