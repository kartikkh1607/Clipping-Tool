@echo off
title Clipper (keep this window open while clips are being made)
cd /d "%~dp0"
set "URL=http://127.0.0.1:7860"

rem Already running? Just open the page.
curl -s -o nul -m 2 %URL%/ && (start "" %URL% & exit /b 0)

call .venv\Scripts\activate.bat
set PYTHONUTF8=1
set GRADIO_ANALYTICS_ENABLED=False
echo Updating yt-dlp...
python -m pip install -q -U "yt-dlp[default]" >nul 2>nul

rem Open the browser as soon as the app answers.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 180;$i++){try{Invoke-WebRequest -UseBasicParsing '%URL%/' -TimeoutSec 1 | Out-Null; Start-Process '%URL%'; break}catch{Start-Sleep -Milliseconds 700}}"

echo Starting Clipper at %URL% ...
python app.py
echo.
echo Clipper stopped.
pause
