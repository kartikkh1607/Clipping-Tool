@echo off
title Clipper setup
cd /d "%~dp0"
echo === Clipper one-time setup ===
echo.
where python >nul 2>nul || (echo Python not found. Install Python 3.11 from python.org and tick "Add to PATH". & pause & exit /b 1)

echo [1/4] Installing FFmpeg...
where ffmpeg >nul 2>nul && (echo FFmpeg already installed.) || winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements

echo [2/4] Installing Deno (needed by yt-dlp for YouTube)...
where deno >nul 2>nul && (echo Deno already installed.) || winget install -e --id DenoLand.Deno --accept-source-agreements --accept-package-agreements

echo [3/4] Creating Python environment...
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [4/4] Installing Python packages (this takes a few minutes)...
pip install -U -r requirements.txt

echo.
echo Setup done. CLOSE this window and open a new one so FFmpeg/Deno are on PATH,
echo then double-click run_clipper.bat
pause
