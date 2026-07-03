# Stap voor stap online zetten

## Stap 0 — Wat heb je nodig?

1. Een GitHub-account.
2. Een Streamlit Community Cloud-account.
3. Een OpenAI API key.
4. Deze map: `digital_video_scout_mvp_v4`.

## Stap 1 — GitHub repository maken

1. Ga naar https://github.com/new
2. Repository name: `digital-video-scout`
3. Kies `Private` als je dit nog niet publiek wil tonen.
4. Klik op `Create repository`.

## Stap 2 — Bestanden uploaden

Makkelijkste route zonder command line:

1. Open je nieuwe GitHub repository.
2. Klik op `Add file` > `Upload files`.
3. Sleep alle bestanden en mappen uit deze map naar GitHub.
4. Upload zeker ook de verborgen map `.streamlit`.
5. Commit message: `Initial cloud app`.
6. Klik op `Commit changes`.

Let op: upload nooit `.env` of `.streamlit/secrets.toml` met een echte key.

## Stap 3 — Streamlit app maken

1. Ga naar https://share.streamlit.io/
2. Klik rechtsboven op `Create app`.
3. Kies `Yup, I have an app`.
4. Selecteer je GitHub-repository.
5. Branch: `main`.
6. Main file path: `app.py`.
7. Kies eventueel een eigen app-url.

## Stap 4 — OpenAI API key als secret plakken

Klik in Streamlit bij deployment op `Advanced settings`.
Plak in het Secrets-vak:

```toml
OPENAI_API_KEY = "sk-jouw-key-hier"
```

Klik op `Save`.

## Stap 5 — Deploy

Klik op `Deploy`.
Wacht een paar minuten.
Daarna krijg je een URL zoals:

```text
https://digital-video-scout.streamlit.app
```

## Stap 6 — Gebruiken

1. Open de Streamlit URL.
2. Kijk links in de sidebar.
3. Laat model staan op `gpt-5.5`.
4. Reasoning: `high`.
5. Vision detail: `high`.
6. Klik links op `🔌 Verbinding maken`.
7. Upload een korte clip.
8. Vul spelernaam, teamkleur en rugnummer in.
9. Klik op `▶ Start analyse`.
10. Download PDF + JSON.

## Veelvoorkomende problemen

### App zegt: geen API key gevonden
Je hebt de secret niet juist geplakt. Ga naar Streamlit app settings > Secrets en plak:

```toml
OPENAI_API_KEY = "sk-..."
```

### Upload is te groot
Begin met een clip van 2 tot 10 minuten. Volledige wedstrijden kunnen traag of te zwaar zijn.

### Model werkt niet
Controleer of jouw OpenAI API-account toegang heeft tot `gpt-5.5`. De app laat toe om links een andere modelnaam in te vullen.

### Analyse is te duur of te traag
Gebruik:
- Frame-interval: 4 à 5 seconden
- Max frames: 40–60
- Frames per batch: 2–4
- Vision detail: low voor test
