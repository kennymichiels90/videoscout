"""
Digital Video Scout MVP v4 - Cloud Ready
---------------------------------------
Streamlit webapp for AI-assisted football video scouting.
Designed for Streamlit Community Cloud so SSL/corporate proxy issues on a work laptop are avoided.

Workflow:
1. Upload match/clip video.
2. Fill player info: team, shirt number, team color, position.
3. Click the sidebar-only "Verbinding maken" button.
4. Start analysis.
5. Download JSON data + PDF report.

Important: this is not a full player-tracking engine. It uses sampled frames + GPT vision.
Human scout validation remains necessary.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

load_dotenv()

APP_NAME = "Digital Video Scout"
APP_VERSION = "v4 Cloud Ready"
DEFAULT_MODEL = "gpt-5.5"


@dataclass
class PlayerConfig:
    player_name: str
    team_name: str
    team_color: str
    shirt_number: str
    position: str
    dominant_foot: str
    report_context: str
    scouting_template: str


@dataclass
class ExtractedFrame:
    index: int
    time_seconds: float
    jpeg_bytes: bytes

    @property
    def timecode(self) -> str:
        total = int(round(self.time_seconds))
        minutes = total // 60
        seconds = total % 60
        return f"{minutes:02d}:{seconds:02d}"


def safe_get_secret(name: str) -> str:
    """Read key from Streamlit secrets, environment, or return empty string."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.getenv(name, "").strip()


def set_page_style() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="⚽", layout="wide")
    st.markdown(
        """
        <style>
        .dvs-hero {
            padding: 28px 30px;
            border-radius: 24px;
            background: linear-gradient(135deg, #06101f 0%, #0b2a4a 55%, #0e5d91 100%);
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 14px 40px rgba(0,0,0,0.28);
            margin-bottom: 22px;
        }
        .dvs-hero h1 { margin: 0; font-size: 2.1rem; letter-spacing: -0.02em; }
        .dvs-hero p { margin: 7px 0 0 0; opacity: 0.88; font-size: 1.02rem; }
        .dvs-card {
            padding: 18px 20px;
            border-radius: 18px;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.10);
            margin-bottom: 14px;
        }
        .dvs-small { opacity: 0.72; font-size: 0.88rem; }
        .stButton > button {
            border-radius: 12px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        f"""
        <div class="dvs-hero">
            <h1>⚽ {APP_NAME}</h1>
            <p>{APP_VERSION} · AI-assisted player reports from match video · Data · Timestamps · PDF</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def response_output_text(response: Any) -> str:
    # Newer SDK exposes output_text; fallback for older/edge responses.
    text = getattr(response, "output_text", None)
    if text:
        return text
    try:
        chunks = []
        for item in response.output:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") in ("output_text", "text"):
                    chunks.append(getattr(content, "text", ""))
        return "\n".join(chunks).strip()
    except Exception:
        return str(response)


def test_openai_connection(api_key: str, model: str, reasoning_effort: str) -> Tuple[bool, str]:
    if not api_key:
        return False, "Geen API key gevonden. Vul OPENAI_API_KEY in bij Streamlit Secrets of plak tijdelijk een key in de sidebar."
    try:
        client = make_openai_client(api_key)
        args: Dict[str, Any] = {
            "model": model,
            "input": "Zeg exact: verbinding werkt",
            "max_output_tokens": 20,
        }
        if reasoning_effort and reasoning_effort != "none":
            args["reasoning"] = {"effort": reasoning_effort}
        response = client.responses.create(**args)
        return True, response_output_text(response) or "Verbonden."
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def resize_image(img: Image.Image, max_width: int) -> Image.Image:
    if img.width <= max_width:
        return img
    ratio = max_width / float(img.width)
    new_size = (max_width, max(1, int(img.height * ratio)))
    return img.resize(new_size)


def encode_pil_as_jpeg(img: Image.Image, max_width: int, quality: int) -> bytes:
    img = img.convert("RGB")
    img = resize_image(img, max_width=max_width)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def extract_frames(video_file: Any, interval_seconds: float, max_frames: int, max_width: int, quality: int) -> Tuple[List[ExtractedFrame], Dict[str, Any]]:
    suffix = Path(video_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise RuntimeError("Video kon niet geopend worden. Gebruik bij voorkeur mp4/h264.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames: List[ExtractedFrame] = []
    t = 0.0
    idx = 1
    while t <= duration and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            jpeg = encode_pil_as_jpeg(pil, max_width=max_width, quality=quality)
            frames.append(ExtractedFrame(index=idx, time_seconds=t, jpeg_bytes=jpeg))
            idx += 1
        t += interval_seconds

    cap.release()
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    metadata = {
        "filename": video_file.name,
        "duration_seconds": round(duration, 1),
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "sampled_frames": len(frames),
        "interval_seconds": interval_seconds,
    }
    return frames, metadata


def jpeg_to_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("utf-8")


def split_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {"raw_text": text, "observations": []}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {"raw_text": text, "observations": []}


def analyze_frame_batch(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    vision_detail: str,
    player: PlayerConfig,
    batch: List[ExtractedFrame],
) -> Dict[str, Any]:
    intro = f"""
Je bent een professionele voetbal-videoscout. Analyseer uitsluitend wat zichtbaar is in de beelden.
Doelspeler:
- Naam: {player.player_name}
- Team: {player.team_name}
- Teamkleur: {player.team_color}
- Rugnummer: {player.shirt_number}
- Positie: {player.position}
- Dominante voet: {player.dominant_foot}

Taak:
1. Herken of de doelspeler zichtbaar is per frame.
2. Noteer alleen acties die redelijk betrouwbaar aan de doelspeler gekoppeld kunnen worden.
3. Geen externe kennis gebruiken. Geen fictieve acties verzinnen.
4. Geef output als geldig JSON-object, zonder markdown.

Gebruik dit schema:
{{
  "batch_summary": "korte samenvatting",
  "observations": [
    {{
      "frame_index": 1,
      "timecode": "00:00",
      "target_visible": true,
      "confidence": "high/medium/low",
      "phase": "in_possession/out_of_possession/transition/set_piece/unknown",
      "action_type": "short_pass/long_pass/carry/dribble/1v1_defending/tackle/interception/aerial_duel/covering_depth/positioning/pressing/shot/cross/other",
      "result": "successful/unsuccessful/neutral/unclear",
      "direction": "progressive/lateral/backward/vertical/diagonal/unclear",
      "club_brugge_principle": "korte link met positieprincipe",
      "detail": "concrete observatie in het Nederlands"
    }}
  ]
}}
""".strip()

    content: List[Dict[str, Any]] = [{"type": "input_text", "text": intro}]
    for fr in batch:
        content.append({"type": "input_text", "text": f"FRAME {fr.index} · timecode {fr.timecode}"})
        content.append({"type": "input_image", "image_url": jpeg_to_data_url(fr.jpeg_bytes), "detail": vision_detail})

    args: Dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 2500,
    }
    if reasoning_effort and reasoning_effort != "none":
        args["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**args)
    return extract_json_object(response_output_text(response))


def aggregate_observations(batch_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for result in batch_results:
        observations.extend(result.get("observations", []) or [])
    # Deduplicate roughly by frame/action/detail
    seen = set()
    unique: List[Dict[str, Any]] = []
    for obs in observations:
        key = (obs.get("frame_index"), obs.get("timecode"), obs.get("action_type"), str(obs.get("detail"))[:80])
        if key not in seen:
            seen.add(key)
            unique.append(obs)
    return unique


def make_counts(observations: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for obs in observations:
        if str(obs.get("target_visible", "")).lower() in ("false", "0", "no"):
            continue
        rows.append({
            "Action type": obs.get("action_type", "unknown"),
            "Result": obs.get("result", "unclear"),
            "Direction": obs.get("direction", "unclear"),
            "Phase": obs.get("phase", "unknown"),
            "Confidence": obs.get("confidence", "unknown"),
        })
    if not rows:
        return pd.DataFrame(columns=["Category", "Value", "Count"])
    df = pd.DataFrame(rows)
    count_rows = []
    for col in ["Action type", "Result", "Direction", "Phase", "Confidence"]:
        vc = df[col].fillna("unknown").value_counts()
        for value, count in vc.items():
            count_rows.append({"Category": col, "Value": value, "Count": int(count)})
    return pd.DataFrame(count_rows)


def generate_final_report(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    player: PlayerConfig,
    video_metadata: Dict[str, Any],
    observations: List[Dict[str, Any]],
    counts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt = f"""
Maak een professioneel Nederlandstalig scoutingsrapport op basis van uitsluitend onderstaande AI-observaties uit video.
Geen externe kennis gebruiken. Maak duidelijk waar data onzeker of beeldbeperkt is.

Spelerconfig:
{json.dumps(asdict(player), ensure_ascii=False, indent=2)}

Videometadata:
{json.dumps(video_metadata, ensure_ascii=False, indent=2)}

Aggregated counts:
{json.dumps(counts, ensure_ascii=False, indent=2)}

Observaties:
{json.dumps(observations, ensure_ascii=False, indent=2)}

Vereiste output als geldig JSON-object:
{{
  "executive_summary": "kort profiel",
  "data_interpretation": "analyse van de data met nuance over betrouwbaarheid",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "club_brugge_fit": "link met football principles voor de positie",
  "scouting_report": "samenhangende concluderende tekst van minstens 300 woorden; start positief en eindig negatief; begin zinnen niet telkens met 'deze'",
  "general_conclusion": "algemene conclusie tussen 50 en 80 woorden",
  "recommendation": "No follow / Rewatch / Keep monitoring / Actively follow / Target",
  "score_out_of_10": "cijfer met korte motivatie"
}}
""".strip()
    args: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 4500,
    }
    if reasoning_effort and reasoning_effort != "none":
        args["reasoning"] = {"effort": reasoning_effort}
    response = client.responses.create(**args)
    return extract_json_object(response_output_text(response))


def pdf_paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    text = (text or "").replace("\n", "<br/>")
    return Paragraph(text, style)


def build_pdf(
    player: PlayerConfig,
    video_metadata: Dict[str, Any],
    observations: List[Dict[str, Any]],
    counts_df: pd.DataFrame,
    final_report: Dict[str, Any],
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.3 * cm, bottomMargin=1.3 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DVSTitle", parent=styles["Title"], textColor=colors.HexColor("#0B2A4A"), fontSize=22, leading=26)
    h2 = ParagraphStyle("DVSH2", parent=styles["Heading2"], textColor=colors.HexColor("#0E5D91"), fontSize=14, leading=18, spaceBefore=10)
    body = ParagraphStyle("DVSBody", parent=styles["BodyText"], fontSize=9.2, leading=12)
    small = ParagraphStyle("DVSSmall", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#444444"))

    story: List[Any] = []
    story.append(pdf_paragraph("Digital Video Scout", title_style))
    story.append(pdf_paragraph(f"{player.player_name} · #{player.shirt_number} · {player.position}", h2))
    story.append(pdf_paragraph(f"Rapportdatum: {datetime.now().strftime('%Y-%m-%d %H:%M')} · Modelanalyse via video frame-sampling", small))
    story.append(Spacer(1, 0.3 * cm))

    meta_rows = [
        ["Speler", player.player_name], ["Team", player.team_name], ["Teamkleur", player.team_color],
        ["Rugnummer", player.shirt_number], ["Positie", player.position], ["Dominante voet", player.dominant_foot],
        ["Video", video_metadata.get("filename", "")], ["Duur", f"{video_metadata.get('duration_seconds', 0)} sec"],
        ["Frames geanalyseerd", str(video_metadata.get("sampled_frames", 0))]
    ]
    t = Table(meta_rows, colWidths=[4.2 * cm, 11 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0B2A4A")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    for key, label in [
        ("executive_summary", "Executive summary"),
        ("data_interpretation", "Data-interpretatie"),
        ("club_brugge_fit", "Link met football principles"),
        ("scouting_report", "Concluderende scoutingsanalyse"),
        ("general_conclusion", "Algemene conclusie"),
        ("recommendation", "Advies"),
        ("score_out_of_10", "Score")
    ]:
        if final_report.get(key):
            story.append(pdf_paragraph(label, h2))
            story.append(pdf_paragraph(str(final_report[key]), body))
            story.append(Spacer(1, 0.15 * cm))

    if final_report.get("strengths"):
        story.append(pdf_paragraph("Sterktes", h2))
        story.append(pdf_paragraph("<br/>".join([f"• {x}" for x in final_report.get("strengths", [])]), body))
    if final_report.get("weaknesses"):
        story.append(pdf_paragraph("Werkpunten", h2))
        story.append(pdf_paragraph("<br/>".join([f"• {x}" for x in final_report.get("weaknesses", [])]), body))

    story.append(PageBreak())
    story.append(pdf_paragraph("Datarapport", title_style))
    if not counts_df.empty:
        rows = [["Categorie", "Waarde", "Aantal"]] + counts_df.astype(str).values.tolist()[:60]
        table = Table(rows, colWidths=[4.5 * cm, 7.0 * cm, 2.5 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2A4A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
    else:
        story.append(pdf_paragraph("Geen voldoende beeldzekere acties geteld.", body))

    story.append(PageBreak())
    story.append(pdf_paragraph("Actielog met timestamps", title_style))
    obs_rows = [["Tijd", "Actie", "Resultaat", "Detail"]]
    for obs in observations[:120]:
        obs_rows.append([
            str(obs.get("timecode", "")),
            str(obs.get("action_type", "")),
            str(obs.get("result", "")),
            str(obs.get("detail", ""))[:220],
        ])
    table = Table(obs_rows, colWidths=[2.0 * cm, 3.0 * cm, 2.5 * cm, 8.5 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2A4A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)

    doc.build(story)
    return buffer.getvalue()


def main() -> None:
    set_page_style()

    st.sidebar.markdown("## 🔌 Verbinding")
    secret_key = safe_get_secret("OPENAI_API_KEY")
    key_source = "Streamlit Secrets / environment" if secret_key else "Nog geen key gevonden"
    st.sidebar.caption(f"Key bron: {key_source}")
    manual_key = ""
    if not secret_key:
        manual_key = st.sidebar.text_input("OpenAI API key", type="password", help="Voor online gebruik: zet dit liever in Streamlit Secrets.")
    api_key = secret_key or manual_key.strip()

    model = st.sidebar.text_input("Model", value=DEFAULT_MODEL, help="Standaard op gpt-5.5. Pas aan als je API-account een andere modelnaam vereist.")
    reasoning_effort = st.sidebar.selectbox("Reasoning", ["high", "medium", "low", "none"], index=0)
    vision_detail = st.sidebar.selectbox("Vision detail", ["high", "low", "auto"], index=0)

    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "connection_message" not in st.session_state:
        st.session_state.connection_message = "Nog niet getest."

    # The connection button appears ONLY here in the sidebar.
    if st.sidebar.button("🔌 Verbinding maken", use_container_width=True):
        with st.sidebar.status("Verbinding testen...", expanded=False):
            ok, msg = test_openai_connection(api_key=api_key, model=model, reasoning_effort=reasoning_effort)
        st.session_state.connected = ok
        st.session_state.connection_message = msg

    if st.session_state.connected:
        st.sidebar.success("Verbonden")
    else:
        st.sidebar.warning("Niet verbonden")
    st.sidebar.caption(st.session_state.connection_message)

    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Analyse-instellingen")
    interval_seconds = st.sidebar.slider("Frame-interval in seconden", 1.0, 10.0, 3.0, 0.5)
    max_frames = st.sidebar.slider("Max frames", 10, 250, 80, 10)
    batch_size = st.sidebar.slider("Frames per GPT-batch", 1, 10, 4, 1)
    max_width = st.sidebar.select_slider("Max beeldbreedte", options=[480, 640, 720, 960, 1280], value=960)
    jpeg_quality = st.sidebar.slider("JPEG kwaliteit", 35, 90, 65, 5)

    hero()
    st.markdown(
        """
        <div class="dvs-card">
            <b>Cloudversie:</b> de OpenAI-verbinding gebeurt online vanaf Streamlit Cloud, niet vanaf je werklaptop. 
            Daardoor omzeil je lokale SSL/proxyproblemen. De knop <b>Verbinding maken</b> staat bewust alleen links in de sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("1. Video")
        video_file = st.file_uploader("Upload wedstrijd of geknipte beelden", type=["mp4", "mov", "m4v", "avi", "mkv"])
        st.caption("Tip: start met 2–10 minuten. Volledige wedstrijden kosten meer tijd en API-budget.")
    with col_b:
        st.subheader("2. Doelspeler")
        player_name = st.text_input("Spelernaam", value="Axel Koukaba")
        team_name = st.text_input("Team", value="Blauw team")
        shirt_number = st.text_input("Rugnummer", value="4")
        team_color = st.text_input("Teamkleur", value="blauw")
        position = st.selectbox("Positie", [
            "Centrale verdediger", "Rechter centrale verdediger", "Linker centrale verdediger", "Wingback", "Flankverdediger",
            "Controlerende middenvelder", "Centrale middenvelder", "Aanvallende middenvelder", "Winger", "Spits", "Doelman", "Andere"
        ], index=0)
        dominant_foot = st.selectbox("Dominante voet", ["Onbekend", "Rechts", "Links", "Beide"], index=0)
        report_context = st.text_input("Rapportcontext", value="Club Brugge scouting")
        scouting_template = st.selectbox("Rapporttemplate", ["Club Brugge stijl", "Korte screening", "Uitgebreid datarapport"], index=0)

    st.subheader("3. Start")
    can_start = st.session_state.connected and video_file is not None and bool(player_name.strip()) and bool(shirt_number.strip())
    if not st.session_state.connected:
        st.info("Klik eerst links in de sidebar op ‘🔌 Verbinding maken’. De startknop blijft geblokkeerd tot dat lukt.")
    start = st.button("▶ Start analyse", type="primary", disabled=not can_start, use_container_width=True)

    if start:
        player = PlayerConfig(
            player_name=player_name.strip(), team_name=team_name.strip(), team_color=team_color.strip(), shirt_number=shirt_number.strip(),
            position=position, dominant_foot=dominant_foot, report_context=report_context.strip(), scouting_template=scouting_template,
        )
        try:
            client = make_openai_client(api_key)
            with st.status("Frames uit video halen...", expanded=True) as status:
                frames, metadata = extract_frames(video_file, interval_seconds, max_frames, max_width, jpeg_quality)
                st.write(f"{len(frames)} frames geëxtraheerd uit {metadata.get('duration_seconds')} seconden video.")
                batches = split_batches(frames, batch_size)
                st.write(f"{len(batches)} GPT-batches voorbereid.")
                status.update(label="Video voorbereid", state="complete")

            batch_results: List[Dict[str, Any]] = []
            progress = st.progress(0)
            for i, batch in enumerate(batches, start=1):
                with st.status(f"Batch {i}/{len(batches)} analyseren...", expanded=False):
                    try:
                        result = analyze_frame_batch(client, model, reasoning_effort, vision_detail, player, batch)
                    except Exception as exc:
                        result = {"batch_summary": f"Batch mislukt: {type(exc).__name__}: {exc}", "observations": [], "error": str(exc)}
                    batch_results.append(result)
                progress.progress(i / len(batches))

            observations = aggregate_observations(batch_results)
            counts_df = make_counts(observations)
            counts_records = counts_df.to_dict(orient="records") if not counts_df.empty else []

            with st.status("Eindrapport genereren...", expanded=False):
                final_report = generate_final_report(client, model, reasoning_effort, player, metadata, observations, counts_records)
                pdf_bytes = build_pdf(player, metadata, observations, counts_df, final_report)

            output = {
                "app_version": APP_VERSION,
                "created_at": datetime.now().isoformat(),
                "player": asdict(player),
                "video_metadata": metadata,
                "settings": {
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "vision_detail": vision_detail,
                    "interval_seconds": interval_seconds,
                    "max_frames": max_frames,
                    "batch_size": batch_size,
                    "max_width": max_width,
                    "jpeg_quality": jpeg_quality,
                },
                "batch_results": batch_results,
                "observations": observations,
                "counts": counts_records,
                "final_report": final_report,
            }

            st.success("Analyse klaar")
            k1, k2, k3 = st.columns(3)
            k1.metric("Frames", metadata.get("sampled_frames", 0))
            k2.metric("Observaties", len(observations))
            k3.metric("Advies", final_report.get("recommendation", "—"))

            st.subheader("Rapport")
            st.write(final_report.get("executive_summary", ""))
            st.markdown("### Concluderende scoutingsanalyse")
            st.write(final_report.get("scouting_report", ""))
            st.markdown("### Algemene conclusie")
            st.write(final_report.get("general_conclusion", ""))

            if not counts_df.empty:
                st.subheader("Datarapport")
                st.dataframe(counts_df, use_container_width=True)

            if observations:
                st.subheader("Actielog")
                st.dataframe(pd.DataFrame(observations), use_container_width=True)

            st.download_button(
                "⬇️ Download PDF-rapport",
                data=pdf_bytes,
                file_name=f"scouting_report_{player.player_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Download JSON-data",
                data=json.dumps(output, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"scouting_data_{player.player_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Analyse mislukt: {type(exc).__name__}: {exc}")
            st.exception(exc)


if __name__ == "__main__":
    main()
