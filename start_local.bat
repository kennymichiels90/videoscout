@echo off
setlocal
cd /d "%~dp0"
echo ===========================================
echo Digital Video Scout MVP v4 - local start
echo ===========================================
echo.
py -m venv .venv
if errorlevel 1 (
  echo FOUT: Python is niet gevonden. Installeer Python en vink Add python.exe to PATH aan.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
pause
