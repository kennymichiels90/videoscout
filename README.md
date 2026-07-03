# Digital Video Scout MVP v6 - Player Lock

AI-assisted football video scouting app for Streamlit Cloud.

## Nieuw in v6

- **Player Lock** vóór de scoutinganalyse.
- Optionele **referentiebeelden/screenshots** van de doelspeler.
- Optionele **duidelijke timestamps** waar de speler zichtbaar is.
- Extra **focus-timestamps** waar de app meer frames rond scant.
- **Confidence-drempel**: high / medium_high / medium.
- **Smart crops/zoom** voor betere herkenning in wide shots.
- Geen volwaardig scoutingsrapport wanneer de speler onvoldoende betrouwbaar herkend wordt, tenzij je dit expliciet forceert.
- PDF bevat nu ook Player Lock data + identity log.

## Hoe werkt het?

1. Upload video.
2. Vul spelerinfo in: naam, teamkleur, rugnummer, positie.
3. Voeg bij voorkeur een screenshot/crop toe waarop de speler duidelijk zichtbaar is.
4. Geef optioneel timestamps waar hij duidelijk zichtbaar is.
5. Klik links in de sidebar op **Verbinding maken**.
6. Klik op **Start Player Lock + analyse**.
7. Download PDF + JSON.

## Streamlit Cloud secrets

Zet je OpenAI API key in Streamlit Secrets:

```toml
OPENAI_API_KEY = "sk-jouw-key-hier"
```

## Aanbevolen eerste instellingen

- Clip: 2-5 minuten
- Regulier frame-interval: 3 sec
- Max reguliere frames: 100
- Identity batch: 4
- Action batch: 3
- Min confidence: medium_high
- Smart crops: aan

## Belangrijke nuance

Dit is nog geen volwaardige player-tracking engine. Het is een AI-assisted MVP. Voor echt professionele workflows is menselijke validatie nodig.
