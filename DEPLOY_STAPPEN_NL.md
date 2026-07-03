# Deployment stappen - Digital Video Scout v6

## 1. Download en unzip

Download `digital_video_scout_mvp_v6.zip` en pak het uit.

## 2. Upload naar GitHub

Upload de inhoud van de map naar je GitHub repository. Upload zeker:

- `app.py`
- `requirements.txt`
- `packages.txt`
- `runtime.txt`
- `.streamlit/config.toml`
- `.streamlit/secrets.toml.example`
- `README.md`
- `DEPLOY_STAPPEN_NL.md`
- `.gitignore`

Upload niet:

- `.env`
- `.venv/`
- video’s
- output/PDF/JSON
- echte `secrets.toml` met API key

## 3. Streamlit Cloud

Ga naar Streamlit Cloud en kies:

- Repository: `jouwnaam/videoscout`
- Branch: `main`
- Main file path: `app.py`

## 4. Secrets invullen

Bij Advanced settings / Secrets:

```toml
OPENAI_API_KEY = "sk-jouw-key-hier"
```

## 5. Deploy

Klik op Deploy of Reboot/Redeploy als je een bestaande app update.

## 6. Update bestaande app

Als je al v5 online hebt:

1. Vervang `app.py` op GitHub door de v6-versie.
2. Vervang `README.md` en `DEPLOY_STAPPEN_NL.md` eventueel mee.
3. Laat `packages.txt` leeg.
4. Commit changes.
5. Ga naar Streamlit Cloud > Manage app > Reboot/Redeploy.

## 7. Gebruik

1. Klik links in de sidebar op **Verbinding maken**.
2. Upload een clip.
3. Vul de spelergegevens in.
4. Voeg bij voorkeur screenshot/crop of duidelijke timestamps toe.
5. Start Player Lock + analyse.
6. Download PDF + JSON.
