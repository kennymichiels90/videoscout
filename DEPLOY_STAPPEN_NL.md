# Deploy-stappen v7 — Streamlit Cloud

## 1. Download en unzip
Pak `digital_video_scout_mvp_v7.zip` uit.

## 2. Upload naar GitHub
Upload de inhoud van de map naar je repo. Niet de zip zelf.

Upload zeker:
- `app.py`
- `requirements.txt`
- `packages.txt`
- `runtime.txt`
- `.streamlit/config.toml`
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

## 6. Deploy / Redeploy
Klik op Deploy of Reboot/Redeploy als je al een app had.

## 7. Test
Open de app en klik links in de sidebar op **Verbinding maken**.

## 8. Eerste analyse
Gebruik eerst:
- korte clip van 2-5 minuten;
- rugnummer correct invullen;
- teamkleur correct invullen;
- eventueel één player-lock screenshot uploaden;
- optionele uiterlijke hint alleen invullen als herkenning moeilijk is.
