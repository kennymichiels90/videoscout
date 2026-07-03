@echo off
setlocal
cd /d "%~dp0"
echo ===========================================
echo Digital Video Scout MVP v4 - GitHub prep
echo ===========================================
echo.
where git >nul 2>nul
if errorlevel 1 (
  echo Git is niet gevonden.
  echo Installeer Git for Windows via https://git-scm.com/download/win
  echo Daarna dit bestand opnieuw dubbelklikken.
  pause
  exit /b 1
)

echo [1/4] Git repository initialiseren...
git init

echo [2/4] Bestanden toevoegen...
git add app.py requirements.txt packages.txt runtime.txt README.md .gitignore .streamlit/config.toml .streamlit/secrets.toml.example start_local.bat prepare_github_repo.bat

echo [3/4] Eerste commit maken...
git commit -m "Initial Digital Video Scout cloud app"

echo.
echo [4/4] Klaar.
echo Maak nu op GitHub.com een nieuwe repository aan, bv. digital-video-scout.
echo Kopieer daarna de commando's die GitHub toont bij 'push an existing repository'.
echo.
echo Ik open nu GitHub en Streamlit Cloud in je browser.
start https://github.com/new
start https://share.streamlit.io/
echo.
pause
