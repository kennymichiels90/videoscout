"""
Digital Video Scout MVP v7 - Contact Lock
-----------------------------------------
Streamlit webapp for AI-assisted football video scouting.

Nieuwe v7 workflow:
1. Upload match/clip video.
2. Fill player info: team, team color, shirt number, position.
3. Optional: add reference screenshots / clear timestamps where the player is visible.
4. Contact Lock identity scan: identify the target player with minimal extra input.
5. Group likely frames into contact/actie-momenten with temporal context.
6. Generate a position-specific PDF report, including the central defender template.

Important: this is still not a full tracking engine. It uses sampled frames + GPT vision.
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
APP_VERSION = "v7 Contact Lock"
DEFAULT_MODEL = "gpt-5.5"
CONFIDENCE_RANK = {"high": 4, "medium_high": 3, "medium-high": 3, "medium": 2, "low": 1, "unknown": 0, "unclear": 0}


CENTRAL_DEFENDER_TEMPLATE_INSTRUCTIONS = """
Gebruik voor een centrale verdediger exact onderstaande overzichtstabelstructuur.
Geef bij elke hoofdcategorie een score op 10, volgens deze schaal:
1-2 = very weak, 3-4 = weak, 5-6 = neutral, 7 = strong, 8 = tier 2, 9 = tier 1, 10 = world class.
Subquotes moeten exact één van deze waarden zijn: Weakness / Neutral / Strength.

Categorieën:
1v1 Defending:
- text
- score_out_of_10
- Agility
- Tackling

Aerial Duels:
- text
- score_out_of_10
- Duel strength
- Heading technique
- Jumping
- Timing

Covering Depth:
- text
- score_out_of_10
- Anticipation
- Body orientation
- Initial acceleration
- Speed

Dynamic Defending:
- text
- score_out_of_10
- Agility
- Close marking
- Duel strength
- Initial acceleration
- Positive aggression in duels

Guiding Defense:
- text
- score_out_of_10
- Coaching
- Leadership
- Winning mentality

Positional Defending:
- text
- score_out_of_10
- Anticipation
- Concentration
- Fall back
- Recognizing when to release opponent
- Recognizing when to support full back
- Split vision

Ball Progression:
- text
- score_out_of_10
- Bravery
- Line breaking passing

Ball Retention:
- text
- score_out_of_10
- Availability
- Composure
- First touch
- Passing short decision
- Passing short execution
- Passing under pressure
- Weak foot usage

Carrying:
- text
- score_out_of_10
- Challenging
- Infiltrations with ball

Long Balls:
- text
- score_out_of_10
- Passing long decision
- Passing long execution
"""

POSITION_HINTS = {
    "Centrale verdediger": "Centrale verdediger; verwacht in laatste lijn, centrale as, restverdediging rond middenlijn en eigen zestien. Focus op duels, covering depth, positional defending, ball retention en ball progression.",
    "Rechter centrale verdediger": "Rechter centrale verdediger; verwacht rechts-centraal in laatste lijn en restverdediging. Focus op timing van uitstappen, rugdekking, ondersteuning rechterflank en progressie via passing/carry.",
    "Linker centrale verdediger": "Linker centrale verdediger; verwacht links-centraal in laatste lijn en restverdediging. Focus op timing van uitstappen, rugdekking, ondersteuning linkerflank en progressie via passing/carry.",
    "Winger": "Winger; verwacht op flank/halfspace. Focus op 1v1, diepgang, carries, laatste actie en pressing na balverlies.",
    "Spits": "Spits; verwacht centraal hoog. Focus op looplijnen, kaats, box presence, pressing en afwerking.",
}


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
    position_hint: str
    appearance_hint: str
    focus_timestamps: str
    anchor_timestamps: str


@dataclass
class ExtractedFrame:
    index: int
    time_seconds: float
    jpeg_bytes: bytes
    source: str = "regular"

    @property
    def timecode(self) -> str:
        total = int(round(self.time_seconds))
        minutes = total // 60
        seconds = total % 60
        return f"{minutes:02d}:{seconds:02d}"


@dataclass
class ReferenceImage:
    label: str
    jpeg_bytes: bytes


def safe_get_secret(name: str) -> str:
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
        .dvs-warning {
            padding: 14px 16px;
            border-radius: 14px;
            background: rgba(250, 204, 21, 0.12);
            border: 1px solid rgba(250, 204, 21, 0.38);
            margin-bottom: 14px;
        }
        .dvs-ok {
            padding: 14px 16px;
            border-radius: 14px;
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.38);
            margin-bottom: 14px;
        }
        .dvs-small { opacity: 0.72; font-size: 0.88rem; }
        .stButton > button { border-radius: 12px; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        f"""
        <div class="dvs-hero">
            <h1>⚽ {APP_NAME}</h1>
            <p>{APP_VERSION} · Contact Lock · AI-assisted scouting reports · Data · Timestamps · PDF</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def response_output_text(response: Any) -> str:
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


def jpeg_to_pil(jpeg_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")


def jpeg_to_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("utf-8")


def parse_timecode_item(raw: str) -> Optional[float]:
    s = raw.strip()
    if not s:
        return None
    s = s.replace(",", ".")
    if ":" not in s:
        try:
            return float(s)
        except Exception:
            return None
    parts = s.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        return None
    return None


def parse_timecodes(raw: str) -> List[float]:
    if not raw:
        return []
    chunks = re.split(r"[;\n,]+", raw)
    values: List[float] = []
    for ch in chunks:
        val = parse_timecode_item(ch)
        if val is not None and val >= 0:
            values.append(val)
    # deduplicate while preserving order
    seen = set()
    unique = []
    for v in values:
        key = round(v, 1)
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique


def open_video_to_temp(video_file: Any) -> Tuple[str, Dict[str, Any]]:
    suffix = Path(video_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError("Video kon niet geopend worden. Gebruik bij voorkeur mp4/h264.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    meta = {
        "filename": video_file.name,
        "duration_seconds": round(duration, 1),
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }
    return tmp_path, meta


def read_frame_at(cap: cv2.VideoCapture, t: float, index: int, source: str, max_width: int, quality: int) -> Optional[ExtractedFrame]:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t) * 1000)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    jpeg = encode_pil_as_jpeg(pil, max_width=max_width, quality=quality)
    return ExtractedFrame(index=index, time_seconds=max(0, t), jpeg_bytes=jpeg, source=source)


def extract_frames_player_lock(
    video_file: Any,
    interval_seconds: float,
    max_regular_frames: int,
    max_width: int,
    quality: int,
    focus_timestamps_raw: str,
    anchor_timestamps_raw: str,
    focus_window_seconds: float,
    focus_interval_seconds: float,
) -> Tuple[List[ExtractedFrame], List[ExtractedFrame], Dict[str, Any]]:
    tmp_path, metadata = open_video_to_temp(video_file)
    duration = float(metadata.get("duration_seconds", 0) or 0)
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise RuntimeError("Video kon niet geopend worden na upload.")

    times: List[Tuple[float, str]] = []
    t = 0.0
    while t <= duration and len([x for x in times if x[1] == "regular"]) < max_regular_frames:
        times.append((t, "regular"))
        t += interval_seconds

    focus_ts = parse_timecodes(focus_timestamps_raw)
    for center in focus_ts:
        start = max(0.0, center - focus_window_seconds)
        end = min(duration, center + focus_window_seconds)
        tt = start
        while tt <= end:
            times.append((tt, "focus"))
            tt += focus_interval_seconds

    anchor_ts = parse_timecodes(anchor_timestamps_raw)
    for center in anchor_ts:
        times.append((min(max(center, 0.0), duration), "anchor"))

    # Deduplicate by half-second; prioritize anchor > focus > regular
    priority = {"regular": 1, "focus": 2, "anchor": 3}
    by_key: Dict[float, Tuple[float, str]] = {}
    for t, src in times:
        key = round(t * 2) / 2
        old = by_key.get(key)
        if old is None or priority[src] > priority[old[1]]:
            by_key[key] = (t, src)

    ordered = sorted(by_key.values(), key=lambda x: x[0])
    frames: List[ExtractedFrame] = []
    anchors: List[ExtractedFrame] = []
    for i, (t, src) in enumerate(ordered, start=1):
        fr = read_frame_at(cap, t, i, src, max_width=max_width, quality=quality)
        if fr:
            frames.append(fr)
            if src == "anchor":
                anchors.append(fr)
    cap.release()
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    metadata.update({
        "sampled_frames": len(frames),
        "regular_interval_seconds": interval_seconds,
        "focus_timestamps": [round(x, 2) for x in focus_ts],
        "anchor_timestamps": [round(x, 2) for x in anchor_ts],
        "anchor_frames": len(anchors),
        "focus_window_seconds": focus_window_seconds,
        "focus_interval_seconds": focus_interval_seconds,
    })
    return frames, anchors, metadata


def load_reference_uploads(files: Optional[List[Any]], max_width: int, quality: int) -> List[ReferenceImage]:
    refs: List[ReferenceImage] = []
    if not files:
        return refs
    for i, file in enumerate(files, start=1):
        try:
            img = Image.open(file).convert("RGB")
            jpeg = encode_pil_as_jpeg(img, max_width=max_width, quality=quality)
            refs.append(ReferenceImage(label=f"uploaded_reference_{i}", jpeg_bytes=jpeg))
        except Exception:
            continue
    return refs[:6]


def frame_as_reference(frames: List[ExtractedFrame], max_items: int = 5) -> List[ReferenceImage]:
    refs: List[ReferenceImage] = []
    for fr in frames[:max_items]:
        refs.append(ReferenceImage(label=f"anchor_{fr.timecode}", jpeg_bytes=fr.jpeg_bytes))
    return refs


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
            return {"raw_text": text, "observations": [], "identity_frames": []}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {"raw_text": text, "observations": [], "identity_frames": []}


def normalize_confidence(value: Any) -> str:
    s = str(value or "unknown").lower().replace(" ", "_").replace("-", "_")
    if s in ("mediumhigh", "medium_high"):
        return "medium_high"
    if s in ("high", "medium", "low"):
        return s
    return "unknown"


def confidence_ok(value: Any, threshold: str) -> bool:
    conf = normalize_confidence(value)
    th = normalize_confidence(threshold)
    return CONFIDENCE_RANK.get(conf, 0) >= CONFIDENCE_RANK.get(th, 3)


def crop_zone(jpeg_bytes: bytes, zone: str, max_width: int, quality: int) -> Optional[bytes]:
    try:
        img = jpeg_to_pil(jpeg_bytes)
        w, h = img.size
        z = (zone or "full").lower()
        if z in ("left", "far_left"):
            box = (0, 0, int(w * 0.48), h)
        elif z in ("center", "centre", "central"):
            box = (int(w * 0.25), 0, int(w * 0.75), h)
        elif z in ("right", "far_right"):
            box = (int(w * 0.52), 0, w, h)
        elif z in ("top", "upper"):
            box = (0, 0, w, int(h * 0.55))
        elif z in ("bottom", "lower"):
            box = (0, int(h * 0.45), w, h)
        else:
            return None
        cropped = img.crop(box)
        return encode_pil_as_jpeg(cropped, max_width=max_width, quality=quality)
    except Exception:
        return None


def build_reference_content(reference_images: List[ReferenceImage]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if reference_images:
        content.append({"type": "input_text", "text": "REFERENTIEBEELDEN VAN DOELSPELER. Gebruik deze alleen om de doelspeler beter te herkennen; analyseer ze niet als wedstrijdactie."})
        for ref in reference_images[:8]:
            content.append({"type": "input_text", "text": f"REFERENCE · {ref.label}"})
            content.append({"type": "input_image", "image_url": jpeg_to_data_url(ref.jpeg_bytes), "detail": "high"})
    return content


def analyze_identity_batch(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    vision_detail: str,
    player: PlayerConfig,
    batch: List[ExtractedFrame],
    reference_images: List[ReferenceImage],
) -> Dict[str, Any]:
    intro = f"""
Je bent een professionele voetbal-videoanalist. Je taak is NIET om al te scouten, maar eerst de doelspeler te herkennen.
Analyseer uitsluitend wat zichtbaar is in de beelden. Gebruik geen externe kennis.

Doelspeler:
- Naam: {player.player_name}
- Team: {player.team_name}
- Teamkleur: {player.team_color}
- Rugnummer: {player.shirt_number}
- Positie: {player.position}
- Dominante voet: {player.dominant_foot}
- Positionele hint: {player.position_hint}
- Uiterlijke/extra hint: {player.appearance_hint}

Strikte regels:
1. Geef alleen 'target_visible=true' als de speler redelijk zichtbaar is op basis van rugnummer, teamkleur, positie, referentiebeelden of duidelijke context.
2. Als het rugnummer niet leesbaar is maar positie/context sterk matcht, gebruik maximaal 'medium' confidence.
3. Gebruik 'medium_high' of 'high' alleen wanneer rugnummer/lichaam/referentie/positie duidelijk overeenkomen.
4. Noteer géén uitgebreide scoutingactie in deze stap. Alleen herkenning + korte context.
5. Geef geldig JSON-object zonder markdown.

Schema:
{{
  "identity_batch_summary": "kort in het Nederlands",
  "identity_frames": [
    {{
      "frame_index": 1,
      "timecode": "00:00",
      "source": "regular/focus/anchor",
      "target_visible": true,
      "confidence": "high/medium_high/medium/low/unknown",
      "approx_zone": "left/center/right/top/bottom/full/unknown",
      "reason": "waarom wel/niet herkend",
      "action_context": "zeer korte context, bv. restverdediging/opbouw/duel/onduidelijk",
      "should_analyze": true
    }}
  ]
}}
""".strip()

    content: List[Dict[str, Any]] = [{"type": "input_text", "text": intro}]
    content.extend(build_reference_content(reference_images))
    content.append({"type": "input_text", "text": "KANDIDAATFRAMES UIT DE VIDEO"})
    for fr in batch:
        content.append({"type": "input_text", "text": f"FRAME {fr.index} · timecode {fr.timecode} · source {fr.source}"})
        content.append({"type": "input_image", "image_url": jpeg_to_data_url(fr.jpeg_bytes), "detail": vision_detail})

    args: Dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 2600,
    }
    if reasoning_effort and reasoning_effort != "none":
        args["reasoning"] = {"effort": reasoning_effort}
    response = client.responses.create(**args)
    return extract_json_object(response_output_text(response))


def aggregate_identity(identity_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for res in identity_results:
        rows.extend(res.get("identity_frames", []) or [])
    seen = set()
    unique: List[Dict[str, Any]] = []
    for row in rows:
        key = (row.get("frame_index"), row.get("timecode"))
        if key in seen:
            continue
        seen.add(key)
        row["confidence"] = normalize_confidence(row.get("confidence"))
        unique.append(row)
    return unique


def select_frames_for_action(frames: List[ExtractedFrame], identity_frames: List[Dict[str, Any]], threshold: str) -> Tuple[List[Tuple[ExtractedFrame, Dict[str, Any]]], Dict[str, Any]]:
    frame_by_idx = {fr.index: fr for fr in frames}
    selected: List[Tuple[ExtractedFrame, Dict[str, Any]]] = []
    counts = {"high": 0, "medium_high": 0, "medium": 0, "low": 0, "unknown": 0, "visible_total": 0, "selected_total": 0}
    for row in identity_frames:
        conf = normalize_confidence(row.get("confidence"))
        counts[conf] = counts.get(conf, 0) + 1
        if bool(row.get("target_visible")):
            counts["visible_total"] += 1
        if bool(row.get("target_visible")) and bool(row.get("should_analyze", True)) and confidence_ok(conf, threshold):
            fr = frame_by_idx.get(int(row.get("frame_index", -1)))
            if fr:
                selected.append((fr, row))
    counts["selected_total"] = len(selected)
    return selected, counts


def analyze_action_batch(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    vision_detail: str,
    player: PlayerConfig,
    batch: List[Tuple[ExtractedFrame, Dict[str, Any]]],
    reference_images: List[ReferenceImage],
    smart_crops: bool,
    crop_max_width: int,
    crop_quality: int,
) -> Dict[str, Any]:
    intro = f"""
Je bent een professionele voetbal-videoscout. Analyseer nu alleen de acties/positionering van de DOELSPELER in de frames die door de Contact Lock zijn geselecteerd.
Gebruik geen externe kennis. Geen fictieve acties verzinnen. Als het toch onzeker is, geef confidence lager en wees eerlijk.

Doelspeler:
- Naam: {player.player_name}
- Team: {player.team_name}
- Teamkleur: {player.team_color}
- Rugnummer: {player.shirt_number}
- Positie: {player.position}
- Dominante voet: {player.dominant_foot}
- Rapportcontext: {player.report_context}
- Positionele hint: {player.position_hint}
- Uiterlijke/extra hint: {player.appearance_hint}

Taak:
1. Noteer alleen acties die aan de doelspeler gekoppeld kunnen worden.
2. Gebruik de identity_confidence uit Contact Lock.
3. Geef per actie een korte, concrete detailobservatie in het Nederlands.
4. Link kort aan een Club Brugge-principe passend bij de positie.
5. Geef geldig JSON-object zonder markdown.

Schema:
{{
  "action_batch_summary": "korte samenvatting",
  "observations": [
    {{
      "frame_index": 1,
      "timecode": "00:00",
      "identity_confidence": "high/medium_high/medium/low",
      "target_location": "left/center/right/top/bottom/full/unknown",
      "phase": "in_possession/out_of_possession/transition/set_piece/unknown",
      "action_type": "short_pass/long_pass/carry/dribble/1v1_defending/tackle/interception/aerial_duel/covering_depth/positioning/pressing/shot/cross/other",
      "result": "successful/unsuccessful/neutral/unclear",
      "direction": "progressive/lateral/backward/vertical/diagonal/unclear",
      "club_brugge_principle": "korte link met positieprincipe",
      "detail": "concrete observatie in het Nederlands",
      "reliability_note": "kort: waarom zeker/onzeker"
    }}
  ]
}}
""".strip()

    content: List[Dict[str, Any]] = [{"type": "input_text", "text": intro}]
    content.extend(build_reference_content(reference_images))
    content.append({"type": "input_text", "text": "GESELECTEERDE FRAMES VOOR ACTIEANALYSE"})
    for fr, idrow in batch:
        conf = normalize_confidence(idrow.get("confidence"))
        zone = str(idrow.get("approx_zone", "unknown"))
        content.append({"type": "input_text", "text": f"FRAME {fr.index} · timecode {fr.timecode} · Contact Lock confidence {conf} · approx_zone {zone} · reason: {idrow.get('reason', '')}"})
        content.append({"type": "input_image", "image_url": jpeg_to_data_url(fr.jpeg_bytes), "detail": vision_detail})
        if smart_crops:
            crop = crop_zone(fr.jpeg_bytes, zone, max_width=crop_max_width, quality=crop_quality)
            if crop:
                content.append({"type": "input_text", "text": f"ZOOM/CROP voor FRAME {fr.index} · zone {zone}"})
                content.append({"type": "input_image", "image_url": jpeg_to_data_url(crop), "detail": "high"})

    args: Dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 3200,
    }
    if reasoning_effort and reasoning_effort != "none":
        args["reasoning"] = {"effort": reasoning_effort}
    response = client.responses.create(**args)
    return extract_json_object(response_output_text(response))


def aggregate_observations(action_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for result in action_results:
        observations.extend(result.get("observations", []) or [])
    seen = set()
    unique: List[Dict[str, Any]] = []
    for obs in observations:
        key = (obs.get("frame_index"), obs.get("timecode"), obs.get("action_type"), str(obs.get("detail"))[:80])
        if key not in seen:
            seen.add(key)
            obs["identity_confidence"] = normalize_confidence(obs.get("identity_confidence"))
            unique.append(obs)
    return unique


def make_counts(observations: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for obs in observations:
        rows.append({
            "Action type": obs.get("action_type", "unknown"),
            "Result": obs.get("result", "unclear"),
            "Direction": obs.get("direction", "unclear"),
            "Phase": obs.get("phase", "unknown"),
            "Identity confidence": obs.get("identity_confidence", "unknown"),
        })
    if not rows:
        return pd.DataFrame(columns=["Category", "Value", "Count"])
    df = pd.DataFrame(rows)
    count_rows = []
    for col in ["Action type", "Result", "Direction", "Phase", "Identity confidence"]:
        vc = df[col].fillna("unknown").value_counts()
        for value, count in vc.items():
            count_rows.append({"Category": col, "Value": value, "Count": int(count)})
    return pd.DataFrame(count_rows)


def identity_quality_text(identity_counts: Dict[str, Any], min_observations: int) -> str:
    selected = int(identity_counts.get("selected_total", 0) or 0)
    if selected >= max(min_observations, 20):
        return "Goed"
    if selected >= min_observations:
        return "Voldoende voor basisrapport"
    if selected > 0:
        return "Te beperkt voor volwaardig rapport"
    return "Onvoldoende"



def seconds_to_timecode(seconds: float) -> str:
    total = int(round(seconds))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"


def build_contact_windows(
    frames: List[ExtractedFrame],
    selected_frames: List[Tuple[ExtractedFrame, Dict[str, Any]]],
    context_seconds: float = 2.0,
    cluster_gap_seconds: float = 3.0,
    max_windows: int = 80,
) -> List[Dict[str, Any]]:
    """Group selected identity frames into short temporal windows.

    This is the key v7 improvement: instead of analyzing isolated stills, the app
    analyzes 3-7 second sequences around moments where the target player was likely visible.
    That is much closer to real contact/action moments.
    """
    if not selected_frames:
        return []
    selected_sorted = sorted(selected_frames, key=lambda x: x[0].time_seconds)
    clusters: List[List[Tuple[ExtractedFrame, Dict[str, Any]]]] = []
    current: List[Tuple[ExtractedFrame, Dict[str, Any]]] = []
    last_t: Optional[float] = None
    for item in selected_sorted:
        t = item[0].time_seconds
        if last_t is None or t - last_t <= cluster_gap_seconds:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
        last_t = t
    if current:
        clusters.append(current)

    windows: List[Dict[str, Any]] = []
    for idx, cluster in enumerate(clusters[:max_windows], start=1):
        start = max(0.0, min(fr.time_seconds for fr, _ in cluster) - context_seconds)
        end = max(fr.time_seconds for fr, _ in cluster) + context_seconds
        window_frames = [fr for fr in frames if start <= fr.time_seconds <= end]
        # keep a reasonable amount of frames per moment
        if len(window_frames) > 9:
            # preserve first, last and evenly sampled middle frames
            step = max(1, len(window_frames) // 8)
            sampled = window_frames[::step][:8]
            if window_frames[-1] not in sampled:
                sampled.append(window_frames[-1])
            window_frames = sampled[:9]
        best_conf = "unknown"
        best_rank = -1
        reasons = []
        zones = []
        for _, row in cluster:
            conf = normalize_confidence(row.get("confidence"))
            rank = CONFIDENCE_RANK.get(conf, 0)
            if rank > best_rank:
                best_rank = rank
                best_conf = conf
            if row.get("reason"):
                reasons.append(str(row.get("reason")))
            if row.get("approx_zone"):
                zones.append(str(row.get("approx_zone")))
        windows.append({
            "moment_id": idx,
            "start_seconds": start,
            "end_seconds": end,
            "start_timecode": seconds_to_timecode(start),
            "end_timecode": seconds_to_timecode(end),
            "best_identity_confidence": best_conf,
            "approx_zone": zones[0] if zones else "unknown",
            "identity_reasons": reasons[:3],
            "selected_frames_in_window": len(cluster),
            "frames": window_frames,
            "selected_identity_rows": [row for _, row in cluster],
        })
    return windows


def analyze_contact_moment_batch(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    vision_detail: str,
    player: PlayerConfig,
    batch: List[Dict[str, Any]],
    reference_images: List[ReferenceImage],
    smart_crops: bool,
    crop_max_width: int,
    crop_quality: int,
) -> Dict[str, Any]:
    intro = f"""
Je bent een professionele voetbal-videoscout. Je analyseert korte videomomenten, opgebouwd uit opeenvolgende frames.
Doel: vind ALLE contactmomenten en relevante actie-/positioneringsmomenten van de doelspeler binnen deze momentvensters.
Gebruik geen externe kennis. Geen fictieve acties verzinnen.

Doelspeler:
- Naam: {player.player_name}
- Team: {player.team_name}
- Teamkleur: {player.team_color}
- Rugnummer: {player.shirt_number}
- Positie: {player.position}
- Dominante voet: {player.dominant_foot}
- Rapportcontext: {player.report_context}
- Positionele hint: {player.position_hint}
- Uiterlijke hint: {player.appearance_hint}

Belangrijke regels:
1. Analyseer de frames als mini-sequentie, niet als losse stills.
2. Noteer een observatie wanneer de doelspeler zichtbaar betrokken is bij balcontact, duel, tackle, luchtduel, onderschepping, pressing, covering depth, restverdediging, rugdekking, positionering of opbouw.
3. Als de doelspeler in het moment wel zichtbaar is maar geen relevante actie/contact heeft, noteer alleen een observatie als zijn positionering tactisch relevant is.
4. Als je twijfelt of de speler echt de doelspeler is, verlaag identity_confidence en zet reliability_note erbij.
5. Vermijd herhaling: één samenhangende actie over 3 frames = één observatie.
6. Geef geldig JSON-object zonder markdown.

Schema:
{{
  "contact_batch_summary": "korte samenvatting",
  "observations": [
    {{
      "moment_id": 1,
      "timecode": "00:00",
      "end_timecode": "00:04",
      "identity_confidence": "high/medium_high/medium/low",
      "target_location": "left/center/right/top/bottom/full/unknown",
      "phase": "in_possession/out_of_possession/transition/set_piece/unknown",
      "involvement_type": "ball_contact/duel_contact/aerial_contact/defensive_position/covering/off_ball_support/pressing/other",
      "action_type": "short_pass/long_pass/carry/dribble/1v1_defending/tackle/interception/aerial_duel/covering_depth/positioning/pressing/shot/cross/other",
      "result": "successful/unsuccessful/neutral/unclear",
      "direction": "progressive/lateral/backward/vertical/diagonal/unclear",
      "club_brugge_principle": "korte link met positieprincipe",
      "detail": "concrete observatie in het Nederlands",
      "reliability_note": "kort: waarom zeker/onzeker"
    }}
  ]
}}
""".strip()
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": intro}]
    content.extend(build_reference_content(reference_images))
    for window in batch:
        content.append({"type": "input_text", "text": f"MOMENT {window['moment_id']} · {window['start_timecode']}-{window['end_timecode']} · best_identity_confidence {window['best_identity_confidence']} · zone {window.get('approx_zone','unknown')} · selected identity frames: {window.get('selected_frames_in_window')} · reasons: {' | '.join(window.get('identity_reasons', [])[:3])}"})
        for fr in window["frames"]:
            content.append({"type": "input_text", "text": f"MOMENT {window['moment_id']} FRAME {fr.index} · timecode {fr.timecode} · source {fr.source}"})
            content.append({"type": "input_image", "image_url": jpeg_to_data_url(fr.jpeg_bytes), "detail": vision_detail})
        if smart_crops and window.get("approx_zone"):
            zone = str(window.get("approx_zone", "unknown"))
            # Add one crop from the central selected frame of the moment, if useful
            frame_list = window.get("frames") or []
            if frame_list:
                mid = frame_list[len(frame_list)//2]
                crop = crop_zone(mid.jpeg_bytes, zone, max_width=crop_max_width, quality=crop_quality)
                if crop:
                    content.append({"type": "input_text", "text": f"ZOOM/CROP MOMENT {window['moment_id']} · zone {zone}"})
                    content.append({"type": "input_image", "image_url": jpeg_to_data_url(crop), "detail": "high"})

    args: Dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 4200,
    }
    if reasoning_effort and reasoning_effort != "none":
        args["reasoning"] = {"effort": reasoning_effort}
    response = client.responses.create(**args)
    return extract_json_object(response_output_text(response))

def generate_final_report(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    player: PlayerConfig,
    video_metadata: Dict[str, Any],
    identity_counts: Dict[str, Any],
    identity_frames: List[Dict[str, Any]],
    observations: List[Dict[str, Any]],
    counts: List[Dict[str, Any]],
    min_observations: int,
    force_report: bool,
) -> Dict[str, Any]:
    quality = identity_quality_text(identity_counts, min_observations)
    prompt = f"""
Maak een professioneel Nederlandstalig scoutingsrapport op basis van uitsluitend onderstaande AI-observaties uit video.
Geen externe kennis gebruiken. Maak duidelijk waar data onzeker of beeldbeperkt is.

BELANGRIJK:
- De app gebruikte v7 Contact Lock. Eerst werd spelerherkenning gedaan, daarna pas actieanalyse.
- Als het aantal selected frames lager is dan minimum en force_report=false, maak dan geen volwaardig positief/negatief scoutingsrapport, maar een 'identificatie onvoldoende'-rapport met rewatch-advies.

Spelerconfig:
{json.dumps(asdict(player), ensure_ascii=False, indent=2)}

Videometadata:
{json.dumps(video_metadata, ensure_ascii=False, indent=2)}

Identity counts:
{json.dumps(identity_counts, ensure_ascii=False, indent=2)}

Identity quality: {quality}
Minimum bruikbare observaties: {min_observations}
Force report: {force_report}

Aggregated counts:
{json.dumps(counts, ensure_ascii=False, indent=2)}

Observaties voor actieanalyse:
{json.dumps(observations, ensure_ascii=False, indent=2)}

Enkele identity frames als context:
{json.dumps(identity_frames[:40], ensure_ascii=False, indent=2)}

Extra template-instructies voor centrale verdediger:
{CENTRAL_DEFENDER_TEMPLATE_INSTRUCTIONS if "verdediger" in player.position.lower() else "Gebruik een positiepassende tabel, maar centrale verdediger-template is niet verplicht."}

Vereiste output als geldig JSON-object:
{{
  "identity_summary": "korte evaluatie van spelerherkenning",
  "executive_summary": "kort profiel",
  "data_interpretation": "analyse van de data met nuance over betrouwbaarheid",
  "overview_table": {{
    "1v1 Defending": {{"score_out_of_10": 6, "text": "...", "Agility": "Neutral", "Tackling": "Neutral"}},
    "Aerial Duels": {{"score_out_of_10": 5, "text": "...", "Duel strength": "Neutral", "Heading technique": "Neutral", "Jumping": "Neutral", "Timing": "Neutral"}},
    "Covering Depth": {{"score_out_of_10": 7, "text": "...", "Anticipation": "Strength", "Body orientation": "Neutral", "Initial acceleration": "Neutral", "Speed": "Neutral"}},
    "Dynamic Defending": {{"score_out_of_10": 6, "text": "...", "Agility": "Neutral", "Close marking": "Neutral", "Duel strength": "Neutral", "Initial acceleration": "Neutral", "Positive aggression in duels": "Neutral"}},
    "Guiding Defense": {{"score_out_of_10": 5, "text": "...", "Coaching": "Neutral", "Leadership": "Neutral", "Winning mentality": "Neutral"}},
    "Positional Defending": {{"score_out_of_10": 6, "text": "...", "Anticipation": "Neutral", "Concentration": "Neutral", "Fall back": "Neutral", "Recognizing when to release opponent": "Neutral", "Recognizing when to support full back": "Neutral", "Split vision": "Neutral"}},
    "Ball Progression": {{"score_out_of_10": 5, "text": "...", "Bravery": "Neutral", "Line breaking passing": "Neutral"}},
    "Ball Retention": {{"score_out_of_10": 5, "text": "...", "Availability": "Neutral", "Composure": "Neutral", "First touch": "Neutral", "Passing short decision": "Neutral", "Passing short execution": "Neutral", "Passing under pressure": "Neutral", "Weak foot usage": "Neutral"}},
    "Carrying": {{"score_out_of_10": 5, "text": "...", "Challenging": "Neutral", "Infiltrations with ball": "Neutral"}},
    "Long Balls": {{"score_out_of_10": 5, "text": "...", "Passing long decision": "Neutral", "Passing long execution": "Neutral"}}
  }},
  "strengths": ["..."],
  "weaknesses": ["..."],
  "club_brugge_fit": "link met football principles voor de positie",
  "scouting_report": "samenhangende concluderende tekst van minstens 300 woorden; start positief en eindig negatief; begin zinnen niet telkens met 'deze'",
  "general_conclusion": "algemene conclusie tussen 50 en 80 woorden",
  "recommendation": "No follow / Rewatch / Keep monitoring / Actively follow / Target",
  "score_out_of_10": "cijfer met korte motivatie",
  "next_steps": ["concrete aanbevelingen om de volgende analyse beter te maken"]
}}
""".strip()
    args: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 5200,
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
    identity_counts: Dict[str, Any],
    identity_frames: List[Dict[str, Any]],
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
    story.append(pdf_paragraph(f"Rapportdatum: {datetime.now().strftime('%Y-%m-%d %H:%M')} · {APP_VERSION} · Contact Lock + video frame-sampling", small))
    story.append(Spacer(1, 0.3 * cm))

    meta_rows = [
        ["Speler", player.player_name], ["Team", player.team_name], ["Teamkleur", player.team_color],
        ["Rugnummer", player.shirt_number], ["Positie", player.position], ["Dominante voet", player.dominant_foot],
        ["Video", video_metadata.get("filename", "")], ["Duur", f"{video_metadata.get('duration_seconds', 0)} sec"],
        ["Frames gescand", str(video_metadata.get("sampled_frames", 0))], ["Contact Lock selected", str(identity_counts.get("selected_total", 0))],
        ["Identity quality", identity_quality_text(identity_counts, 8)]
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


    # Position-specific overview table, especially for central defenders.
    overview = final_report.get("overview_table")
    if isinstance(overview, dict) and overview:
        story.append(pdf_paragraph("Overzichtstabel", title_style))
        for cat, data in overview.items():
            if not isinstance(data, dict):
                continue
            score = data.get("score_out_of_10", "")
            text = data.get("text", "")
            story.append(pdf_paragraph(f"{cat} — {score}/10", h2))
            if text:
                story.append(pdf_paragraph(str(text), body))
            trait_rows = [["Subcategorie", "Beoordeling"]]
            for k, v in data.items():
                if k in ("text", "score_out_of_10"):
                    continue
                trait_rows.append([str(k), str(v)])
            if len(trait_rows) > 1:
                tbl = Table(trait_rows, colWidths=[7.0 * cm, 7.0 * cm])
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2A4A")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.append(tbl)
            story.append(Spacer(1, 0.18 * cm))
        story.append(PageBreak())

    for key, label in [
        ("identity_summary", "Contact Lock / herkenning"),
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
    if final_report.get("next_steps"):
        story.append(pdf_paragraph("Volgende analyse verbeteren", h2))
        story.append(pdf_paragraph("<br/>".join([f"• {x}" for x in final_report.get("next_steps", [])]), body))

    story.append(PageBreak())
    story.append(pdf_paragraph("Contact Lock data", title_style))
    lock_rows = [["Metric", "Waarde"]] + [[k, str(v)] for k, v in identity_counts.items()]
    table = Table(lock_rows, colWidths=[6.0 * cm, 8.0 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2A4A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.4 * cm))

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
    obs_rows = [["Tijd", "Actie", "Resultaat", "Confidence", "Detail"]]
    for obs in observations[:120]:
        obs_rows.append([
            str(obs.get("timecode", "")),
            str(obs.get("action_type", "")),
            str(obs.get("result", "")),
            str(obs.get("identity_confidence", "")),
            str(obs.get("detail", ""))[:190],
        ])
    if len(obs_rows) == 1:
        obs_rows.append(["—", "—", "—", "—", "Geen actieframes boven confidence-drempel."])
    table = Table(obs_rows, colWidths=[1.7 * cm, 2.6 * cm, 2.1 * cm, 2.4 * cm, 7.2 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2A4A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.1),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)

    story.append(PageBreak())
    story.append(pdf_paragraph("Herkenningslog", title_style))
    id_rows = [["Tijd", "Visible", "Confidence", "Zone", "Reason"]]
    for row in identity_frames[:120]:
        id_rows.append([
            str(row.get("timecode", "")),
            str(row.get("target_visible", "")),
            str(row.get("confidence", "")),
            str(row.get("approx_zone", "")),
            str(row.get("reason", ""))[:190],
        ])
    table = Table(id_rows, colWidths=[1.7 * cm, 2.0 * cm, 2.5 * cm, 2.1 * cm, 7.7 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2A4A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.0),
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
    st.sidebar.markdown("## 🧲 Contact Lock")
    confidence_threshold = st.sidebar.selectbox("Min. confidence voor contactanalyse", ["high", "medium_high", "medium"], index=2)
    min_observations = st.sidebar.slider("Min. herkenningsframes voor rapport", 3, 30, 5, 1)
    force_report = st.sidebar.checkbox("Beperkt rapport toelaten bij weinig herkenning", value=True)
    smart_crops = st.sidebar.checkbox("Smart crops/zoom meesturen", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Analyse-instellingen")
    interval_seconds = st.sidebar.slider("Frame-interval voor herkenning", 0.5, 5.0, 1.0, 0.5)
    max_frames = st.sidebar.slider("Max frames scannen", 20, 500, 180, 10)
    identity_batch_size = st.sidebar.slider("Frames per identity batch", 1, 8, 4, 1)
    action_batch_size = st.sidebar.slider("Contactmomenten per batch", 1, 5, 2, 1)
    focus_window_seconds = st.sidebar.slider("Advanced focuswindow", 2.0, 20.0, 6.0, 1.0)
    focus_interval_seconds = st.sidebar.slider("Focus frame-interval", 0.5, 3.0, 1.0, 0.5)
    max_width = st.sidebar.select_slider("Max beeldbreedte", options=[480, 640, 720, 960, 1280], value=960)
    jpeg_quality = st.sidebar.slider("JPEG kwaliteit", 35, 90, 68, 5)

    hero()
    st.markdown(
        """
        <div class="dvs-card">
            <b>Nieuw in v7:</b> de app werkt met <b>Contact Lock</b>. Eerst wordt de doelspeler herkend, daarna worden de herkenningen automatisch gegroepeerd tot korte contact-/actiemomenten. Zo analyseert de app niet langer losse stilstaande beelden, maar mini-sequenties rond de speler.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("1. Video")
        video_file = st.file_uploader("Upload wedstrijd of geknipte beelden", type=["mp4", "mov", "m4v", "avi", "mkv"])
        st.caption("Tip: start met 2–10 minuten. Upload liefst één duidelijk player-lock beeld of screenshot als het rugnummer moeilijk leesbaar is.")
        reference_uploads = st.file_uploader(
            "Optioneel: upload één duidelijk player-lock beeld van de doelspeler",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="Bij voorkeur crop/screenshot met rugnummer zichtbaar. Eén goed beeld is vaak genoeg.",
        )
    with col_b:
        st.subheader("2. Doelspeler")
        player_name = st.text_input("Spelernaam", value="Harryl Mboma")
        team_name = st.text_input("Team", value="PSG")
        shirt_number = st.text_input("Rugnummer", value="4")
        team_color = st.text_input("Teamkleur", value="blauw")
        position = st.selectbox("Positie", [
            "Centrale verdediger", "Rechter centrale verdediger", "Linker centrale verdediger", "Wingback", "Flankverdediger",
            "Controlerende middenvelder", "Centrale middenvelder", "Aanvallende middenvelder", "Winger", "Spits", "Doelman", "Andere"
        ], index=0)
        dominant_foot = st.selectbox("Dominante voet", ["Onbekend", "Rechts", "Links", "Beide"], index=0)
        report_context = st.text_input("Rapportcontext", value="Club Brugge scouting")
        scouting_template = st.selectbox("Rapporttemplate", ["Club Brugge stijl", "Korte screening", "Uitgebreid datarapport"], index=0)

    st.subheader("3. Herkenning")
    st.caption("Hou dit simpel: rugnummer + teamkleur + eventueel één player-lock beeld of korte uiterlijke hint. De rest gebeurt automatisch.")
    appearance_hint = st.text_area(
        "Optionele uiterlijke hint",
        value="",
        placeholder="Bijv. linksvoetig, lange mouwen, gele schoenen, centrale verdediger links/rechts in de lijn...",
        height=80,
        help="Niet verplicht. Gebruik dit alleen als rugnummers moeilijk leesbaar zijn."
    )
    position_hint = POSITION_HINTS.get(position, f"Positie: {position}. Gebruik teamkleur, rugnummer en positiecontext om de doelspeler te herkennen.")
    with st.expander("Geavanceerd: alleen gebruiken als herkenning moeilijk blijft"):
        anchor_timestamps = st.text_area(
            "Player-lock timestamps, optioneel",
            value="",
            placeholder="Bijv. 00:12, 01:24, 02:33",
            help="Alleen invullen als je zelf al momenten kent waar de speler duidelijk zichtbaar is.",
            height=80,
        )
        focus_timestamps = st.text_area(
            "Focus-timestamps, optioneel",
            value="",
            placeholder="Bijv. 00:30, 01:15, 03:48",
            help="Alleen invullen als je een paar contactmomenten extra wil laten scannen.",
            height=80,
        )
        custom_position_hint = st.text_area(
            "Positionele hint aanpassen, optioneel",
            value=position_hint,
            height=80,
        )
        if custom_position_hint.strip():
            position_hint = custom_position_hint.strip()

    st.subheader("4. Start")
    can_start = st.session_state.connected and video_file is not None and bool(player_name.strip()) and bool(shirt_number.strip())
    if not st.session_state.connected:
        st.info("Klik eerst links in de sidebar op ‘🔌 Verbinding maken’. De startknop blijft geblokkeerd tot dat lukt.")
    start = st.button("▶ Start Contact Lock + analyse", type="primary", disabled=not can_start, use_container_width=True)

    if start:
        player = PlayerConfig(
            player_name=player_name.strip(), team_name=team_name.strip(), team_color=team_color.strip(), shirt_number=shirt_number.strip(),
            position=position, dominant_foot=dominant_foot, report_context=report_context.strip(), scouting_template=scouting_template,
            position_hint=position_hint.strip(), appearance_hint=appearance_hint.strip(), focus_timestamps=focus_timestamps.strip(), anchor_timestamps=anchor_timestamps.strip(),
        )
        try:
            client = make_openai_client(api_key)
            with st.status("Video voorbereiden en frames extraheren...", expanded=True) as status:
                frames, anchor_frames, metadata = extract_frames_player_lock(
                    video_file=video_file,
                    interval_seconds=interval_seconds,
                    max_regular_frames=max_frames,
                    max_width=max_width,
                    quality=jpeg_quality,
                    focus_timestamps_raw=focus_timestamps,
                    anchor_timestamps_raw=anchor_timestamps,
                    focus_window_seconds=focus_window_seconds,
                    focus_interval_seconds=focus_interval_seconds,
                )
                uploaded_refs = load_reference_uploads(reference_uploads, max_width=max_width, quality=jpeg_quality)
                reference_images = uploaded_refs + frame_as_reference(anchor_frames, max_items=5)
                st.write(f"{len(frames)} frames geëxtraheerd uit {metadata.get('duration_seconds')} seconden video.")
                st.write(f"{len(reference_images)} referentiebeelden beschikbaar voor Contact Lock.")
                identity_batches = split_batches(frames, identity_batch_size)
                st.write(f"{len(identity_batches)} identity-batches voorbereid.")
                status.update(label="Video voorbereid", state="complete")

            st.subheader("Contact Lock")
            identity_results: List[Dict[str, Any]] = []
            progress = st.progress(0)
            for i, batch in enumerate(identity_batches, start=1):
                with st.status(f"Identity batch {i}/{len(identity_batches)} analyseren...", expanded=False):
                    try:
                        result = analyze_identity_batch(client, model, reasoning_effort, vision_detail, player, batch, reference_images)
                    except Exception as exc:
                        result = {"identity_batch_summary": f"Batch mislukt: {type(exc).__name__}: {exc}", "identity_frames": [], "error": str(exc)}
                    identity_results.append(result)
                progress.progress(i / len(identity_batches))

            identity_frames = aggregate_identity(identity_results)
            selected_frames, identity_counts = select_frames_for_action(frames, identity_frames, confidence_threshold)
            quality = identity_quality_text(identity_counts, min_observations)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Frames gescand", len(frames))
            k2.metric("Target zichtbaar", identity_counts.get("visible_total", 0))
            k3.metric("Geselecteerd", identity_counts.get("selected_total", 0))
            k4.metric("Identity quality", quality)

            if identity_frames:
                st.dataframe(pd.DataFrame(identity_frames), use_container_width=True)

            enough_data = int(identity_counts.get("selected_total", 0) or 0) >= min_observations
            if not enough_data and not force_report:
                st.markdown(
                    """
                    <div class="dvs-warning">
                        <b>Te weinig betrouwbare spelerherkenning.</b><br/>
                        Er wordt géén volwaardig scoutingsrapport gemaakt. Voeg duidelijke screenshots/timestamps toe, verlaag eventueel de confidence-drempel of gebruik een betere clip.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            observations: List[Dict[str, Any]] = []
            action_results: List[Dict[str, Any]] = []
            counts_df = pd.DataFrame(columns=["Category", "Value", "Count"])
            counts_records: List[Dict[str, Any]] = []

            contact_windows: List[Dict[str, Any]] = []
            if enough_data or force_report:
                st.subheader("Contactmoment-analyse")
                contact_windows = build_contact_windows(
                    frames=frames,
                    selected_frames=selected_frames,
                    context_seconds=2.0,
                    cluster_gap_seconds=3.0,
                    max_windows=80,
                )
                st.write(f"{len(contact_windows)} contact-/actiemomenten opgebouwd rond de herkende speler.")
                action_batches = split_batches(contact_windows, action_batch_size)
                action_progress = st.progress(0)
                for i, batch in enumerate(action_batches, start=1):
                    with st.status(f"Contactmoment batch {i}/{len(action_batches)} analyseren...", expanded=False):
                        try:
                            result = analyze_contact_moment_batch(
                                client, model, reasoning_effort, vision_detail, player, batch, reference_images,
                                smart_crops=smart_crops, crop_max_width=max_width, crop_quality=jpeg_quality,
                            )
                        except Exception as exc:
                            result = {"contact_batch_summary": f"Batch mislukt: {type(exc).__name__}: {exc}", "observations": [], "error": str(exc)}
                        action_results.append(result)
                    action_progress.progress(i / max(len(action_batches), 1))

                observations = aggregate_observations(action_results)
                counts_df = make_counts(observations)
                counts_records = counts_df.to_dict(orient="records") if not counts_df.empty else []

            with st.status("Eindrapport genereren...", expanded=False):
                final_report = generate_final_report(
                    client, model, reasoning_effort, player, metadata, identity_counts, identity_frames,
                    observations, counts_records, min_observations=min_observations, force_report=force_report,
                )
                pdf_bytes = build_pdf(player, metadata, identity_counts, identity_frames, observations, counts_df, final_report)

            output = {
                "app_version": APP_VERSION,
                "created_at": datetime.now().isoformat(),
                "player": asdict(player),
                "video_metadata": metadata,
                "settings": {
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "vision_detail": vision_detail,
                    "confidence_threshold": confidence_threshold,
                    "min_observations": min_observations,
                    "force_report": force_report,
                    "smart_crops": smart_crops,
                    "regular_interval_seconds": interval_seconds,
                    "max_regular_frames": max_frames,
                    "identity_batch_size": identity_batch_size,
                    "action_batch_size": action_batch_size,
                    "max_width": max_width,
                    "jpeg_quality": jpeg_quality,
                },
                "identity_counts": identity_counts,
                "identity_results": identity_results,
                "identity_frames": identity_frames,
                "selected_frame_count": len(selected_frames),
                "contact_window_count": len(contact_windows) if "contact_windows" in locals() else 0,
                "action_results": action_results,
                "observations": observations,
                "counts": counts_records,
                "final_report": final_report,
            }

            st.success("Analyse klaar")
            st.subheader("Rapport")
            st.write(final_report.get("identity_summary", ""))
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
