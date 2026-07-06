# Deploy-stappen v15 — Streamlit Cloud

## 1. Download en unzip
Pak `digital_video_scout_mvp_v15.zip` uit.

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

## 3. Controleer uploadlimiet
In `.streamlit/config.toml` moet staan:

```toml
[server]
maxUploadSize = 3000
enableXsrfProtection = true
```

Dat geeft een uploadlimiet van ongeveer 3 GB.

## 4. Commit changes
Klik onderaan GitHub op **Commit changes**.

## 5. Streamlit Cloud
Open Streamlit Community Cloud en kies:
- Repository: jouw GitHub repo
- Branch: `main`
- Main file path: `app.py`

## 6. Secrets
Voeg in Streamlit Secrets toe:

```toml
OPENAI_API_KEY = "sk-jouw-key-hier"
```

## 7. Reboot / Redeploy
Klik in Streamlit op **Manage app** en daarna **Reboot** of **Redeploy**.

## 8. Gebruik
1. Klik links op **Verbinding maken**.
2. Upload video.
3. Kies **Screening**, **Scouting** of **Team Screening**.
4. Vul de spelerinput of teamline-ups in.
5. Klik op Start.

## Team Screening advies
Gebruik liefst line-ups, bijvoorbeeld:

```text
4 Harryl Mboma - Centrale verdediger
6 Naam Speler - Middenvelder
9 Naam Speler - Spits
```

Zonder line-up kan de app wel rugnummers/teamkleuren proberen herkennen, maar namen koppelen wordt dan minder betrouwbaar.

## Kostenadvies
Begin met **Screening** of **Team Screening** in budgetinstellingen. Gebruik **Scouting** pas voor spelers die uit de screening interessant lijken.


## Belangrijk voor 3 GB upload

Zorg dat `.streamlit/config.toml` in de **root** van je GitHub-repository staat, naast `app.py`. Niet in een extra map zoals `digital_video_scout_mvp_v15/.streamlit/config.toml`.

Controleer in de app onder Video of er staat: `Uploadlimiet ingesteld op 3000 MB / 3 GB`. Als de app nog 200 MB toont: doe in Streamlit Cloud `Manage app → Reboot`.


## Nieuw in v15
- Topbanner opgeschoond: enkel logo en titel.
- Geen extra navigatie/marketingtekst bovenaan.
- Videobron kan nu ook via Google Drive-link of directe videolink.
- Uploadlimiet blijft 3000 MB / 3 GB via `.streamlit/config.toml`.


## Drive-fix in v15
- Google Drive-links worden eerst via `gdown` gedownload.
- Grote Drive-bestanden met bevestigingspagina worden beter ondersteund.
- Het bestand moet gedeeld zijn met: Iedereen met de link kan bekijken.


## Nieuw in v15
- Wit Club Brugge-logo geïntegreerd in de app (`assets/club_brugge_logo.png`).
- Google Drive-fix en 3 GB uploadlimiet blijven behouden.
