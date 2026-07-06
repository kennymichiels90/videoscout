# Club Brugge Video Scout — Club Style

Streamlit-app voor AI-assisted voetbalvideoanalyse in een Club Brugge-geïnspireerde stijl.

## Workflows

1. **Screening**  
   Budgetmodus voor één speler: korte samenhangende tekst + algemene conclusie.

2. **Scouting**  
   Individueel rapport met Player Lock, contactmomenten, positie-template en PDF.

3. **Team Screening**  
   Volledige wedstrijdscan met korte screening per speler, shortlist en rewatch-advies.

## Grote video’s tot ongeveer 3 GB

De app ondersteunt twee manieren:

- **Videolink**: aanbevolen voor grote wedstrijden. De app downloadt de video server-side en verwerkt daarna de frames. Dit omzeilt de 200 MB browser-uploadlimiet.
- **Upload**: handig voor clips. Voor uploads moet `.streamlit/config.toml` in de root van je GitHub-repo staan.

## Videolink-tips

Gebruik bij voorkeur:

- directe `.mp4` / `.mov` downloadlink;
- Dropbox-link met `dl=1`;
- Google Drive-link waarbij delen staat op “Iedereen met de link kan bekijken”.

## Secrets

Zet je OpenAI API key in Streamlit Cloud bij Secrets:

```toml
OPENAI_API_KEY = "sk-..."
```

## Bestanden in de root van GitHub

```text
app.py
requirements.txt
packages.txt
runtime.txt
README.md
DEPLOY_STAPPEN_NL.md
.streamlit/config.toml
assets/club_brugge_logo.png
assets/club_brugge_bg.png
```

Laat `packages.txt` leeg.
