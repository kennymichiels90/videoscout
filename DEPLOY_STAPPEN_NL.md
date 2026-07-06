# Deploy-stappen — Streamlit Cloud

## 1. Zip uitpakken

Pak de zip uit op je computer.

## 2. Upload naar GitHub

Zet de bestanden rechtstreeks in de root van je repository:

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

Niet in een extra map plaatsen.

## 3. OpenAI key toevoegen

In Streamlit Cloud:

```text
Manage app → Settings → Secrets
```

Plak:

```toml
OPENAI_API_KEY = "sk-..."
```

## 4. Reboot

Na elke wijziging aan `.streamlit/config.toml`:

```text
Manage app → Reboot
```

Niet alleen rerun.

## 5. Grote video’s

Als upload toch 200 MB blijft tonen, gebruik in de app **Videolink**. Dan downloadt de app de video server-side en wordt de browser-uploadlimiet vermeden.
