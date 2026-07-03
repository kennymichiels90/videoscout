# Digital Video Scout MVP v7 — Contact Lock

AI-assisted football video scouting app built with Streamlit + OpenAI API.

## Belangrijkste verbetering in v7

v7 probeert de speler niet alleen op losse frames te herkennen. De app werkt nu met **Contact Lock**:

1. Video uploaden.
2. Spelergegevens invullen: naam, team, teamkleur, rugnummer, positie.
3. Optioneel: één duidelijke screenshot/crop uploaden als player-lock beeld of één korte uiterlijke hint invullen.
4. De app scant de video op herkenning van de doelspeler.
5. Herkenningsframes worden automatisch gegroepeerd tot korte contact-/actiemomenten.
6. Die mini-sequenties worden geanalyseerd.
7. De PDF bevat data, actielog, herkenningslog en een positiegericht rapport.

## Gebruiksvriendelijker

De hoofdinterface vraagt nu alleen:
- video;
- spelernaam;
- team;
- teamkleur;
- rugnummer;
- positie;
- optioneel één player-lock beeld;
- optioneel één uiterlijke hint.

Timestamps en extra hints staan in een geavanceerde expander en zijn niet nodig voor normale werking.

## Centrale verdediger-template

Voor centrale verdedigers gebruikt de app nu een vaste overzichtstabel met:
- 1v1 Defending;
- Aerial Duels;
- Covering Depth;
- Dynamic Defending;
- Guiding Defense;
- Positional Defending;
- Ball Progression;
- Ball Retention;
- Carrying;
- Long Balls.

Elke hoofdcategorie krijgt een score op 10 volgens:
1-2 very weak, 3-4 weak, 5-6 neutral, 7 strong, 8 tier 2, 9 tier 1, 10 world class.

## Deploy

Upload de inhoud van deze map naar GitHub en deploy via Streamlit Community Cloud.

Belangrijke bestanden:
- `app.py`
- `requirements.txt`
- `packages.txt` leeg laten
- `runtime.txt`
- `.streamlit/config.toml`

Zet je API-key in Streamlit Secrets:

```toml
OPENAI_API_KEY = "sk-..."
```

## Kosten beperken

Start met korte clips van 2-5 minuten.
Gebruik standaard:
- frame-interval 1 sec;
- max frames 180;
- confidence medium;
- gpt-5.5 high alleen voor kwaliteitsanalyse.

Voor goedkope tests kan je model tijdelijk wijzigen naar een goedkoper model dat beschikbaar is in jouw API-account.
