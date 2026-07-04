# Deploy-stappen v9 — Streamlit Cloud

## 1. Download en unzip
Pak `digital_video_scout_mvp_v9.zip` uit.

## 2. Upload naar GitHub
Upload de inhoud van de map naar je repo. Niet de zip zelf.

Upload zeker:
- `app.py`
- `requirements.txt`
- `packages.txt`
- `runtime.txt`
- `.streamlit/config.toml`
- `.streamlit/secrets.toml.example`
- `README.md`
- `DEPLOY_STAPPEN_NL.md`

Upload niet:
- `.env`
- `.venv/`
- video’s
- rapporten
- echte `secrets.toml` met API-key

## 3. Commit changes
Klik onderaan GitHub op **Commit changes**.

## 4. Streamlit Cloud
Open Streamlit Community Cloud en kies:
- Repository: jouw GitHub repo
- Branch: `main`
- Main file path: `app.py`

## 5. Secrets
Voeg in Streamlit Secrets toe:

```toml
OPENAI_API_KEY = "sk-jouw-key-hier"
```

## 6. Reboot / Redeploy
Klik in Streamlit op **Manage app** en daarna **Reboot** of **Redeploy**.

## 7. Gebruik
1. Klik links op **Verbinding maken**.
2. Upload video.
3. Vul spelernaam, team, teamkleur, rugnummer en positie in.
4. Voeg optioneel één player-lock beeld toe.
5. Kies **Screening** of **Scouting**.
6. Klik op Start.

## Kostenadvies
Begin altijd met **Screening** om goedkoop te testen. Gebruik **Scouting** pas wanneer de speler goed genoeg zichtbaar is.
