# Club Brugge Video Scout MVP v12 — Team Screening

AI-assisted football video scouting app built with Streamlit + OpenAI API.

## Nieuw in v12

v12 voegt een derde module toe: **Team Screening**.

De app heeft nu drie workflows:

### 1. Screening
Budgetmodus voor een eerste snelle beoordeling van één speler.

Output:
- één samenhangende screeningsanalyse;
- algemene conclusie voor Club Brugge;
- korte score/advies;
- lagere kost door minder frames en lichtere instellingen.

### 2. Scouting
Standardmodus voor een volwaardiger individueel scoutingsrapport.

Output:
- Contact Lock + contactmomentanalyse;
- datarapport;
- actielog;
- PDF;
- positiegericht rapport.

Voor centrale verdedigers gebruikt de app de vaste template met:
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

Elke hoofdcategorie krijgt een score op 10:
- 1-2 = very weak
- 3-4 = weak
- 5-6 = neutral
- 7 = strong
- 8 = tier 2
- 9 = tier 1
- 10 = world class

Subcategorieën worden gequote als:
- Weakness
- Neutral
- Strength

### 3. Team Screening
Budgetmodus voor een volledige wedstrijd. De app scant brede wedstrijdbeelden en maakt per speler een korte eerste screening.

Output:
- overzicht per speler;
- aantal gevonden momenten;
- betrouwbaarheid per speler;
- korte screening per speler;
- algemene conclusie;
- shortlist;
- rewatch-lijst;
- PDF + JSON.

Belangrijk: Team Screening is geen vervanging voor Wyscout/Hudl of handmatige scouting. Het is een **shortlist-generator**: wie viel op, wie moet herbekeken worden, en voor wie is een individueel scoutingrapport nuttig?

## Interface

De hoofdinterface is vereenvoudigd:

1. **Video** uploaden
2. **Optie** kiezen
   - Screening
   - Scouting
   - Team Screening
3. **Input**
   - bij Screening/Scouting: spelernaam, team, teamkleur, rugnummer, positie + optioneel player-lock beeld/uiterlijke hint;
   - bij Team Screening: Team A, Team B, teamkleuren en line-up per team.
4. **Start**

## Video upload

De uploadlimiet staat op **3000 MB / 3 GB** via:

```toml
[server]
maxUploadSize = 3000
```

Bestand: `.streamlit/config.toml`

## Kosten

Standaard gebruikt v12 `gpt-5-mini` om de kosten lager te houden.

Richtlijn:
- Screening = goedkoopste individuele modus;
- Team Screening = budget voor brede wedstrijdscan;
- Scouting = duurder, maar vollediger;
- dure modellen zijn niet standaard nodig en kunnen manueel in de sidebar worden gekozen.

## Deploy

Upload deze bestanden naar GitHub:
- app.py
- requirements.txt
- packages.txt
- runtime.txt
- README.md
- DEPLOY_STAPPEN_NL.md
- .streamlit/config.toml
- .streamlit/secrets.toml.example

Laat `packages.txt` leeg.

Zet je OpenAI key in Streamlit Secrets:

```toml
OPENAI_API_KEY = "sk-jouw-key-hier"
```
