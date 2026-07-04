# Digital Video Scout MVP v9 — Screening & Scouting

AI-assisted football video scouting app built with Streamlit + OpenAI API.

## Nieuw in v9

v9 maakt een duidelijk onderscheid tussen twee workflows:

### 1. Screening
Budgetmodus voor een eerste snelle beoordeling.

Output:
- één samenhangende screeningsanalyse;
- algemene conclusie voor Club Brugge;
- korte score/advies;
- lagere kost door minder frames en lichtere instellingen.

### 2. Scouting
Standardmodus voor een volwaardiger scoutingsrapport.

Output:
- Contact Lock + contactmomentanalyse;
- datarapport;
- actielog;
- PDF;
- positiegericht rapport.

Voor centrale verdedigers gebruikt de app de vaste template met onder meer:
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

## Interface

De hoofdinterface is vereenvoudigd:

1. Video uploaden
2. Speler invullen
   - spelernaam;
   - team;
   - teamkleur;
   - rugnummer;
   - positie;
   - optioneel player-lock beeld;
   - optioneel uiterlijke hint.
3. Optie kiezen
   - Screening
   - Scouting
4. Start

De velden Dominante voet, Rapportcontext en Rapporttemplate zijn uit de hoofdinterface gehaald.

## Kosten

Standaard gebruikt v9 `gpt-5-mini` om de kosten lager te houden.

Richtlijn:
- Screening = goedkoopste modus;
- Scouting = duurder, maar vollediger;
- gpt-5.5 is niet standaard meer nodig en kan eventueel manueel in de sidebar worden gekozen.

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
