# Digital Video Scout MVP v4 — Cloud Ready

Deze versie is bedoeld om online te draaien via Streamlit Community Cloud. Daardoor loopt de OpenAI-verbinding niet meer via je werk-laptop of bedrijfsproxy, wat SSL/certificaatproblemen meestal omzeilt.

## Belangrijkste wijzigingen in v4

- De knop **🔌 Verbinding maken** staat **enkel links in de sidebar**.
- De app is klaar voor online deployment via Streamlit Community Cloud.
- De OpenAI API key wordt automatisch gelezen uit **Streamlit Secrets**.
- Er staat geen echte API key in de code.
- `packages.txt`, `runtime.txt`, `.streamlit/config.toml` en `.gitignore` zijn toegevoegd voor cloud-deployment.
- De app blijft lokaal ook werken, maar de aanbevolen route is online.

## Bestanden

- `app.py` — Streamlit-app
- `requirements.txt` — Python dependencies
- `packages.txt` — Linux packages voor cloud
- `runtime.txt` — Python-versie voor cloud
- `.streamlit/config.toml` — thema en upload-instellingen
- `.streamlit/secrets.toml.example` — voorbeeld van secrets
- `.gitignore` — voorkomt dat keys/video's mee naar GitHub gaan
- `prepare_github_repo.bat` — maakt lokaal automatisch een Git-repo en commit aan
- `start_local.bat` — lokale start, optioneel

## Online deployment in 1 zin

Upload deze map naar GitHub, ga naar Streamlit Community Cloud, kies `app.py`, plak je OpenAI API key in Secrets, en deploy.

## Streamlit Secrets

Plak in Streamlit Cloud > Advanced settings > Secrets:

```toml
OPENAI_API_KEY = "sk-..."
```

Nooit je API key hardcoden in `app.py` of committen naar GitHub.
