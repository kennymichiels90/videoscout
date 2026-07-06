"""
Digital Video Scout MVP v10 - Team Screening
-----------------------------------------
Streamlit webapp for AI-assisted football video scouting.

Nieuwe v10 workflow:
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
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pandas as pd
import streamlit as st

UPLOAD_LIMIT_MB = 3000

def large_file_uploader(*args, **kwargs):
    """Use Streamlit's per-widget max_upload_size when available, fallback to global config."""
    kwargs.setdefault("max_upload_size", UPLOAD_LIMIT_MB)
    try:
        return st.file_uploader(*args, **kwargs)
    except TypeError:
        # Older Streamlit versions don't support max_upload_size yet.
        kwargs.pop("max_upload_size", None)
        return st.file_uploader(*args, **kwargs)

def active_upload_limit_mb():
    try:
        return int(st.config.get_option("server.maxUploadSize"))
    except Exception:
        return None
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
APP_VERSION = "v13 Drive Link Clean Style"
DEFAULT_MODEL = "gpt-5-mini"
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


def asset_data_uri(path: str) -> str:
    try:
        mime = "image/png"
        suffix = Path(path).suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        data = Path(path).read_bytes()
        return f"data:{mime};base64," + base64.b64encode(data).decode("utf-8")
    except Exception:
        return ""


def set_page_style() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="⚽", layout="wide")
    bg_uri = asset_data_uri("assets/club_brugge_bg.png")
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        :root {{
            --cb-black: #060b14;
            --cb-navy: #0a111d;
            --cb-navy-2: #0d1726;
            --cb-card: #0b1523;
            --cb-card-2: #101d30;
            --cb-blue: #0877ff;
            --cb-blue-2: #1292ff;
            --cb-blue-dark: #004ea8;
            --cb-text: #edf5ff;
            --cb-muted: #95a8c1;
            --cb-line: rgba(8,119,255,0.38);
            --cb-white-line: rgba(238,245,255,0.12);
        }}

        html, body, .stApp {{
            background:
                radial-gradient(circle at 85% 0%, rgba(8,119,255,0.14) 0%, rgba(8,119,255,0.02) 34%, transparent 58%),
                linear-gradient(135deg, rgba(255,255,255,0.035) 0 9%, transparent 9% 18%, rgba(255,255,255,0.022) 18% 27%, transparent 27% 100%),
                #060b14 !important;
            color: var(--cb-text) !important;
            font-family: 'Inter', sans-serif !important;
        }}

        .main .block-container {{
            max-width: 1180px;
            padding-top: 0.75rem;
            padding-bottom: 3rem;
        }}

        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 0.035em;
            color: var(--cb-text) !important;
            font-weight: 700 !important;
        }}

        h1 {{ font-size: 4.8rem !important; line-height: 0.9 !important; }}
        h2 {{ font-size: 2.05rem !important; margin-top: 1.2rem !important; }}
        h3 {{ font-size: 1.45rem !important; }}

        p, label, span, div, li {{
            color: var(--cb-text);
        }}

        .dvs-mini-header {{
            position: relative;
            min-height: 150px;
            margin: -0.75rem calc(50% - 50vw) 2.0rem calc(50% - 50vw);
            padding: 18px calc(50vw - 590px) 24px calc(50vw - 590px);
            background:
                linear-gradient(180deg, rgba(6,11,20,0.40) 0%, rgba(6,11,20,0.78) 66%, rgba(6,11,20,1) 100%),
                url('{bg_uri}') center 42% / cover no-repeat;
            border-bottom: 1px solid rgba(238,245,255,0.10);
            overflow: hidden;
        }}

        .dvs-mini-header::after {{
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(125deg, rgba(255,255,255,0.04) 0 12%, transparent 12% 26%, rgba(8,119,255,0.07) 26% 43%, transparent 43% 100%);
            pointer-events: none;
        }}

        .dvs-nav {{
            position: relative;
            z-index: 2;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 52px;
        }}

        .dvs-nav-left, .dvs-nav-right {{
            display: flex;
            align-items: center;
            gap: 26px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.045em;
            font-size: 0.95rem;
        }}

        .dvs-menu {{
            font-size: 1.35rem;
            opacity: 0.92;
        }}

        .dvs-nav-link.active {{
            color: var(--cb-blue) !important;
        }}

        .dvs-nav-logo {{
            position: absolute;
            left: 50%;
            top: 2px;
            transform: translateX(-50%);
            width: 58px;
            height: 58px;
            object-fit: contain;
            filter: drop-shadow(0 8px 16px rgba(0,0,0,0.45));
        }}

        .dvs-nav-btn {{
            border: 1px solid rgba(238,245,255,0.85);
            border-radius: 999px;
            padding: 13px 22px;
            background: rgba(255,255,255,0.02);
        }}

        .dvs-nav-btn.primary {{
            background: var(--cb-blue);
            color: #05101f !important;
            border-color: var(--cb-blue);
        }}

        .dvs-title-row {{
            position: relative;
            z-index: 2;
            margin-top: 34px;
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 20px;
        }}

        .dvs-title {{
            font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
            font-size: 4.4rem;
            line-height: 0.88;
            font-weight: 800;
            letter-spacing: 0.035em;
            text-transform: uppercase;
            color: #edf5ff;
            text-shadow: 0 10px 24px rgba(0,0,0,0.42);
        }}

        .dvs-mode-pills {{
            display: flex;
            border: 1px solid var(--cb-blue);
            border-radius: 999px;
            overflow: hidden;
            background: rgba(6,11,20,0.72);
            margin-bottom: 8px;
        }}

        .dvs-mode-pills span {{
            padding: 13px 20px;
            color: var(--cb-blue) !important;
            border-right: 1px solid var(--cb-blue);
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.04em;
        }}

        .dvs-mode-pills span:first-child {{
            background: var(--cb-blue);
            color: #06101f !important;
        }}

        .dvs-mode-pills span:last-child {{
            border-right: none;
        }}

        .dvs-card {{
            padding: 18px 20px;
            border-radius: 0;
            background: var(--cb-card);
            border: 1px solid var(--cb-white-line);
            border-left: 3px solid var(--cb-blue);
            box-shadow: none;
            margin-bottom: 14px;
        }}

        .dvs-card b, .dvs-card strong {{
            color: #ffffff !important;
        }}

        .dvs-warning {{
            padding: 14px 16px;
            border-radius: 0;
            background: #182009;
            border: 1px solid rgba(255,194,0,0.40);
            color: #ffe08a !important;
            margin-bottom: 14px;
        }}

        .dvs-ok {{
            padding: 14px 16px;
            border-radius: 0;
            background: #071f16;
            border: 1px solid rgba(50,220,140,0.36);
            color: #beffd9 !important;
            margin-bottom: 14px;
        }}

        .dvs-small {{
            color: var(--cb-muted) !important;
            font-size: 0.88rem;
        }}

        [data-testid="stSidebar"] {{
            background: #070e19 !important;
            border-right: 1px solid rgba(238,245,255,0.09);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--cb-text) !important;
        }}

        .stButton > button {{
            border-radius: 999px !important;
            min-height: 46px !important;
            background: var(--cb-blue) !important;
            color: #05101f !important;
            border: 1px solid var(--cb-blue) !important;
            box-shadow: none !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            font-weight: 800 !important;
        }}

        .stButton > button:hover {{
            background: var(--cb-blue-2) !important;
            border-color: var(--cb-blue-2) !important;
            color: #05101f !important;
        }}

        .stButton > button:disabled {{
            background: #1b2636 !important;
            border-color: #2c3a4f !important;
            color: #7d8ea7 !important;
        }}

        .stDownloadButton > button {{
            border-radius: 999px !important;
            background: transparent !important;
            color: var(--cb-blue) !important;
            border: 1px solid var(--cb-blue) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            font-weight: 800 !important;
        }}

        input, textarea {{
            color: #edf5ff !important;
            background: #0c1727 !important;
            caret-color: #edf5ff !important;
        }}

        .stTextInput input, .stTextArea textarea, .stNumberInput input {{
            background: #0c1727 !important;
            border: 1px solid rgba(238,245,255,0.16) !important;
            border-radius: 0 !important;
            color: #edf5ff !important;
        }}

        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {{
            background: #0c1727 !important;
            border: 1px solid rgba(238,245,255,0.16) !important;
            border-radius: 0 !important;
            color: #edf5ff !important;
        }}

        div[data-baseweb="popover"], div[data-baseweb="menu"] {{
            background: #0c1727 !important;
            color: #edf5ff !important;
        }}

        div[data-baseweb="option"] {{
            background: #0c1727 !important;
            color: #edf5ff !important;
        }}

        div[data-baseweb="option"]:hover {{
            background: #102542 !important;
        }}

        .stRadio > div {{
            gap: 10px;
        }}

        .stRadio label {{
            background: #0c1727 !important;
            color: #edf5ff !important;
            border: 1px solid rgba(238,245,255,0.16) !important;
            border-radius: 999px !important;
            padding: 10px 18px !important;
        }}

        .stRadio label:has(input:checked) {{
            border-color: var(--cb-blue) !important;
            background: rgba(8,119,255,0.22) !important;
        }}

        div[data-testid="stFileUploader"] {{
            background: transparent !important;
        }}

        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploaderDropzone"] {{
            background: #0c1727 !important;
            border: 1px dashed var(--cb-line) !important;
            border-radius: 0 !important;
            color: #edf5ff !important;
        }}

        div[data-testid="stFileUploader"] section *,
        div[data-testid="stFileUploaderDropzone"] * {{
            color: #edf5ff !important;
        }}

        div[data-testid="stFileUploader"] button {{
            background: transparent !important;
            color: var(--cb-blue) !important;
            border: 1px solid var(--cb-blue) !important;
            border-radius: 999px !important;
        }}

        .stAlert {{
            background: #101a2a !important;
            color: #edf5ff !important;
            border: 1px solid rgba(238,245,255,0.13) !important;
            border-radius: 0 !important;
        }}

        .stAlert * {{
            color: #edf5ff !important;
        }}

        .stExpander {{
            background: #0b1523 !important;
            border: 1px solid rgba(238,245,255,0.12) !important;
            border-radius: 0 !important;
        }}

        div[data-testid="stMetric"] {{
            background: #0b1523 !important;
            border: 1px solid rgba(238,245,255,0.10);
            border-left: 3px solid var(--cb-blue);
            padding: 12px 14px;
        }}

        .stDataFrame, .stTable {{
            background: #0b1523 !important;
            color: #edf5ff !important;
            border: 1px solid rgba(238,245,255,0.10);
        }}

        [data-testid="stMarkdownContainer"] a {{
            color: var(--cb-blue) !important;
        }}

        hr {{
            border-color: rgba(238,245,255,0.12) !important;
        }}

        @media (max-width: 900px) {{
            .dvs-mini-header {{
                padding-left: 18px;
                padding-right: 18px;
                min-height: 170px;
            }}
            .dvs-nav-left .dvs-nav-link, .dvs-nav-right {{
                display: none;
            }}
            .dvs-title-row {{
                margin-top: 40px;
                align-items: flex-start;
                flex-direction: column;
            }}
            .dvs-title {{
                font-size: 3.4rem;
            }}
            .dvs-mode-pills span {{
                padding: 10px 12px;
                font-size: 0.84rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )




def hero() -> None:
    logo_uri = asset_data_uri("assets/club_brugge_logo.png")
    st.markdown(
        f"""
        <div class="dvs-mini-header">
            <div class="dvs-nav">
                <img class="dvs-nav-logo" src="{logo_uri}" alt="Club Brugge logo" />
            </div>
            <div class="dvs-title-row">
                <div class="dvs-title">Video Scout</div>
            </div>
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


def google_drive_direct_url(url: str) -> str:
    """Convert common Google Drive share links to a direct download URL."""
    raw = (url or "").strip()
    if not raw:
        return raw
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/open\?id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            file_id = match.group(1)
            return f"https://drive.google.com/uc?export=download&id={file_id}"
    return raw


def download_video_url_to_temp(url: str, max_mb: int = UPLOAD_LIMIT_MB) -> Tuple[str, str]:
    """Download a public/direct video link to a temp file. Works best with public Google Drive links."""
    direct_url = google_drive_direct_url(url)
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name

    session = requests.Session()
    response = session.get(direct_url, stream=True, timeout=60)
    # Google Drive sometimes asks a virus-scan confirmation token for large files.
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token:
        sep = "&" if "?" in direct_url else "?"
        response.close()
        response = session.get(f"{direct_url}{sep}confirm={token}", stream=True, timeout=60)

    response.raise_for_status()
    downloaded = 0
    max_bytes = max_mb * 1024 * 1024
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > max_bytes:
                response.close()
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                raise RuntimeError(f"Video is groter dan {max_mb} MB.")
            f.write(chunk)

    # Quick check: Google Drive error pages are HTML, not video.
    try:
        with open(tmp_path, "rb") as f:
            head = f.read(512).lower()
        if b"<html" in head or b"<!doctype html" in head:
            raise RuntimeError("De link leverde een HTML-pagina op in plaats van een videobestand. Zet de Google Drive-link op 'Iedereen met de link kan bekijken' of gebruik een directe downloadlink.")
    except RuntimeError:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

    return tmp_path, direct_url


def inspect_video_file(tmp_path: str, filename: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise RuntimeError("Video kon niet geopend worden. Gebruik bij voorkeur mp4/h264.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "filename": filename,
        "duration_seconds": round(duration, 1),
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def open_video_source_to_temp(video_file: Any = None, video_url: str = "") -> Tuple[str, Dict[str, Any]]:
    if video_file is not None:
        suffix = Path(video_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        metadata = inspect_video_file(tmp_path, video_file.name)
        metadata["source"] = "upload"
        return tmp_path, metadata

    if video_url and video_url.strip():
        tmp_path, direct_url = download_video_url_to_temp(video_url.strip())
        metadata = inspect_video_file(tmp_path, "google_drive_or_url_video.mp4")
        metadata["source"] = "link"
        metadata["video_url"] = video_url.strip()
        metadata["download_url_used"] = direct_url
        return tmp_path, metadata

    raise RuntimeError("Geen video-upload of videolink gevonden.")


def open_video_to_temp(video_file: Any) -> Tuple[str, Dict[str, Any]]:
    # Backwards-compatible wrapper for older parts of the app.
    return open_video_source_to_temp(video_file=video_file, video_url="")




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
    video_file: Any = None,
    video_url: str = "",
    interval_seconds: float = 3.0,
    max_regular_frames: int = 120,
    max_width: int = 960,
    quality: int = 65,
    focus_timestamps_raw: str = "",
    anchor_timestamps_raw: str = "",
    focus_window_seconds: float = 6.0,
    focus_interval_seconds: float = 1.0,
) -> Tuple[List[ExtractedFrame], List[ExtractedFrame], Dict[str, Any]]:
    tmp_path, metadata = open_video_source_to_temp(video_file=video_file, video_url=video_url)
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
    report_type: str,
) -> Dict[str, Any]:
    """Generate either a light screening report or a full scouting report.

    report_type:
      - "Screening": 1 cohesive text + general conclusion
      - "Scouting": position template, e.g. central defender table + conclusion
    """
    quality = identity_quality_text(identity_counts, min_observations)
    report_type_norm = (report_type or "Screening").strip().lower()

    base_context = f"""
Maak een professioneel Nederlandstalig rapport op basis van uitsluitend onderstaande AI-observaties uit video.
Geen externe kennis gebruiken. Maak duidelijk waar data onzeker of beeldbeperkt is.

BELANGRIJK:
- De app gebruikt v10 Contact Lock. Eerst wordt de doelspeler gezocht/herkend, daarna worden alleen contact-/actiemomenten geanalyseerd.
- Als de speler onvoldoende betrouwbaar herkend is, wees eerlijk en maak geen schijnzeker rapport.
- Begin bij evaluaties steeds met positieve punten en eindig met negatieve punten.
- Verwerk de voetbalprincipes van Club Brugge waar relevant voor de positie: hoog durven verdedigen, restverdediging bewaken, vooruit verdedigen, balvastheid onder druk, progressie zoeken, intensiteit na balverlies en dominante/gerichte duels.
- Begin zinnen niet telkens met 'deze'.

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
""".strip()

    if report_type_norm.startswith("screen"):
        prompt = base_context + """

RAPPORTTYPE: SCREENINGSRAPPORT / BUDGETMODUS.
Doel: een eerste, snelle Club Brugge-screening. Geen lange template en geen uitgebreide categorieën.

Vereiste output als geldig JSON-object:
{
  "report_type": "Screening",
  "identity_summary": "korte evaluatie van spelerherkenning",
  "executive_summary": "maximaal 4 zinnen over profiel en betrouwbaarheid",
  "data_interpretation": "korte data-interpretatie met nuance over beeldzekerheid",
  "screening_text": "één samenhangende concluderende tekst voor een scoutingsrapport van 220 tot 320 woorden. Begin positief, eindig negatief. Verwerk Club Brugge-principes en positieprincipes. Vermeld duidelijk als de beelden onvoldoende betrouwbaar zijn.",
  "general_conclusion": "algemene conclusie tussen 50 en 80 woorden",
  "recommendation": "No follow / Rewatch / Keep monitoring / Actively follow / Target",
  "score_out_of_10": "cijfer met korte motivatie",
  "next_steps": ["concrete aanbeveling"]
}
""".strip()
        max_tokens = 2800
    else:
        prompt = base_context + f"""

RAPPORTTYPE: SCOUTINGSRAPPORT / STANDARDMODUS.
Gebruik voor een centrale verdediger exact onderstaande template. Als de positie geen centrale verdediger is, pas de principes aan, maar behoud de structuur waar mogelijk.

{CENTRAL_DEFENDER_TEMPLATE_INSTRUCTIONS}

Vereiste output als geldig JSON-object:
{{
  "report_type": "Scouting",
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
  "scouting_report": "samenhangende concluderende tekst van minstens 300 woorden. Begin positief en eindig negatief. Verwerk Club Brugge-principes en de positieprincipes.",
  "general_conclusion": "algemene conclusie tussen 50 en 80 woorden",
  "recommendation": "No follow / Rewatch / Keep monitoring / Actively follow / Target",
  "score_out_of_10": "cijfer met korte motivatie",
  "next_steps": ["concrete aanbeveling"]
}}
""".strip()
        max_tokens = 5200

    args: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
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
        ["Rugnummer", player.shirt_number], ["Positie", player.position],
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
        ("screening_text", "Screeningsanalyse"),
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



# -----------------------------------------------------------------------------
# v10 Team Screening / Shortlist Generator
# -----------------------------------------------------------------------------

def parse_lineup_text(raw: str, team_name: str, team_color: str) -> List[Dict[str, str]]:
    """Parse a simple pasted lineup.

    Supported examples:
      4 Harryl Mboma - Centrale verdediger
      #6 John Doe, CM
      9; Player Name; Spits
      Player Name #10 Winger
    """
    players: List[Dict[str, str]] = []
    if not raw:
        return players
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("-") and len(s) < 3:
            continue
        s = re.sub(r"^[•\-\*]\s*", "", s)
        number = ""
        name = ""
        position = "Onbekend"

        if ";" in s:
            parts = [p.strip() for p in s.split(";") if p.strip()]
            if len(parts) >= 2:
                # either nr; name; position or name; nr; position
                if re.fullmatch(r"#?\d{1,2}", parts[0]):
                    number = parts[0].replace("#", "")
                    name = parts[1]
                    if len(parts) >= 3:
                        position = parts[2]
                elif len(parts) >= 2 and re.fullmatch(r"#?\d{1,2}", parts[1]):
                    name = parts[0]
                    number = parts[1].replace("#", "")
                    if len(parts) >= 3:
                        position = parts[2]
        if not name:
            # #4 Harryl Mboma - CV  OR  4 Harryl Mboma, CV
            m = re.match(r"^#?(\d{1,2})\s+(.+?)(?:\s+[-–—,]\s+(.+))?$", s)
            if m:
                number = m.group(1)
                name = m.group(2).strip()
                if m.group(3):
                    position = m.group(3).strip()
        if not name:
            # Harryl Mboma #4 - CV
            m = re.match(r"^(.+?)\s+#?(\d{1,2})(?:\s+[-–—,]\s+(.+))?$", s)
            if m:
                name = m.group(1).strip()
                number = m.group(2)
                if m.group(3):
                    position = m.group(3).strip()
        if name:
            name = re.sub(r"\s+", " ", name).strip(" -–—,")
            players.append({
                "team": team_name.strip() or "Onbekend team",
                "team_color": team_color.strip() or "onbekend",
                "shirt_number": number.strip(),
                "player_name": name,
                "position": position.strip() or "Onbekend",
            })
    # de-duplicate on team + number/name
    unique: List[Dict[str, str]] = []
    seen = set()
    for p in players:
        key = (p.get("team", "").lower(), p.get("shirt_number", ""), p.get("player_name", "").lower())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def lineup_to_prompt(players: List[Dict[str, str]]) -> str:
    if not players:
        return "Geen line-up opgegeven. Rapporteer dan op rugnummer/teamkleur, maar koppel geen namen als je ze niet kent."
    return json.dumps(players, ensure_ascii=False, indent=2)


def extract_frames_team_screening(
    video_file: Any = None,
    video_url: str = "",
    interval_seconds: float = 6.0,
    max_regular_frames: int = 220,
    max_width: int = 720,
    quality: int = 55,
) -> Tuple[List[ExtractedFrame], Dict[str, Any]]:
    frames, _anchors, metadata = extract_frames_player_lock(
        video_file=video_file,
        video_url=video_url,
        interval_seconds=interval_seconds,
        max_regular_frames=max_regular_frames,
        max_width=max_width,
        quality=quality,
        focus_timestamps_raw="",
        anchor_timestamps_raw="",
        focus_window_seconds=3.0,
        focus_interval_seconds=1.0,
    )
    metadata["team_screening_mode"] = True
    return frames, metadata


def analyze_team_frame_batch(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    vision_detail: str,
    batch: List[ExtractedFrame],
    team_a_name: str,
    team_a_color: str,
    team_b_name: str,
    team_b_color: str,
    lineups: List[Dict[str, str]],
    scan_scope: str,
) -> Dict[str, Any]:
    intro = f"""
Je bent een professionele voetbal-videoanalist. Je analyseert een wedstrijdscan met als doel per speler korte screeningsrapporten mogelijk te maken.
Gebruik geen externe kennis. Baseer je uitsluitend op de zichtbare frames.

Teams:
- Team A: {team_a_name} · kleur: {team_a_color}
- Team B: {team_b_name} · kleur: {team_b_color}
Scan scope: {scan_scope}

Line-up / spelerslijst:
{lineup_to_prompt(lineups)}

Taak in deze stap:
1. Detecteer per frame welke spelers duidelijk of waarschijnlijk zichtbaar/betrokken zijn.
2. Koppel waar mogelijk aan team + rugnummer + naam uit de line-up.
3. Noteer alleen momenten met duidelijke betrokkenheid: balcontact, duel, luchtduel, tackle, interceptie, pressing, loopactie, positioneel relevant moment of duidelijke verdedigende/offensieve actie.
4. Als naam of rugnummer onzeker is: gebruik player_name='Onbekend' en zet reliability lager.
5. Geen lange teksten. Alleen compacte observaties in JSON.
6. Vermijd fictie: als een rugnummer niet leesbaar is, zeg dat.

Geldig JSON-object zonder markdown:
{{
  "batch_summary": "kort in het Nederlands",
  "player_moments": [
    {{
      "timecode": "00:00",
      "team": "teamnaam",
      "team_color": "kleur",
      "shirt_number": "4/of onbekend",
      "player_name": "naam uit line-up of Onbekend",
      "position": "positie of onbekend",
      "identity_confidence": "high/medium_high/medium/low/unknown",
      "involvement_type": "ball_contact/duel_contact/aerial_contact/defensive_position/off_ball_run/pressing/support/other",
      "action_type": "short_pass/long_pass/carry/dribble/1v1_defending/tackle/interception/aerial_duel/covering_depth/positioning/pressing/shot/cross/other",
      "result": "successful/unsuccessful/neutral/unclear",
      "detail": "één concrete observatie in het Nederlands",
      "reliability_note": "waarom zeker/onzeker"
    }}
  ]
}}
""".strip()
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": intro}]
    for fr in batch:
        content.append({"type": "input_text", "text": f"FRAME {fr.index} · timecode {fr.timecode} · source {fr.source}"})
        content.append({"type": "input_image", "image_url": jpeg_to_data_url(fr.jpeg_bytes), "detail": vision_detail})

    args: Dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 3600,
    }
    if reasoning_effort and reasoning_effort != "none":
        args["reasoning"] = {"effort": reasoning_effort}
    response = client.responses.create(**args)
    return extract_json_object(response_output_text(response))


def aggregate_team_moments(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    moments: List[Dict[str, Any]] = []
    for res in results:
        moments.extend(res.get("player_moments", []) or [])
    unique: List[Dict[str, Any]] = []
    seen = set()
    for m in moments:
        m["identity_confidence"] = normalize_confidence(m.get("identity_confidence"))
        key = (
            str(m.get("timecode", "")),
            str(m.get("team", "")).lower(),
            str(m.get("shirt_number", "")),
            str(m.get("player_name", "")).lower(),
            str(m.get("action_type", "")),
            str(m.get("detail", ""))[:70],
        )
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def player_key(moment: Dict[str, Any]) -> str:
    team = str(moment.get("team", "Onbekend team")).strip() or "Onbekend team"
    nr = str(moment.get("shirt_number", "")).strip()
    name = str(moment.get("player_name", "Onbekend")).strip() or "Onbekend"
    if nr and nr.lower() not in ("onbekend", "unknown"):
        return f"{team} #{nr} {name}"
    return f"{team} {name}"


def summarize_team_moments(moments: List[Dict[str, Any]], lineups: List[Dict[str, str]]) -> pd.DataFrame:
    # Ensure all lineup players appear, even when no moments were detected.
    base_keys: Dict[str, Dict[str, Any]] = {}
    for p in lineups:
        k = f"{p.get('team','Onbekend team')} #{p.get('shirt_number','')} {p.get('player_name','Onbekend')}".strip()
        base_keys[k] = {
            "Speler": p.get("player_name", "Onbekend"),
            "Team": p.get("team", ""),
            "#": p.get("shirt_number", ""),
            "Positie": p.get("position", "Onbekend"),
            "Momenten": 0,
            "High/med-high": 0,
            "Balcontact/duel": 0,
            "Betrouwbaarheid": "onvoldoende",
        }
    for m in moments:
        k = player_key(m)
        if k not in base_keys:
            base_keys[k] = {
                "Speler": m.get("player_name", "Onbekend"),
                "Team": m.get("team", ""),
                "#": m.get("shirt_number", ""),
                "Positie": m.get("position", "Onbekend"),
                "Momenten": 0,
                "High/med-high": 0,
                "Balcontact/duel": 0,
                "Betrouwbaarheid": "onvoldoende",
            }
        base_keys[k]["Momenten"] += 1
        conf = normalize_confidence(m.get("identity_confidence"))
        if CONFIDENCE_RANK.get(conf, 0) >= CONFIDENCE_RANK.get("medium_high", 3):
            base_keys[k]["High/med-high"] += 1
        inv = str(m.get("involvement_type", "")).lower()
        act = str(m.get("action_type", "")).lower()
        if "contact" in inv or act in ("short_pass", "long_pass", "carry", "dribble", "tackle", "interception", "aerial_duel", "shot", "cross"):
            base_keys[k]["Balcontact/duel"] += 1
    for row in base_keys.values():
        n = int(row["Momenten"])
        hi = int(row["High/med-high"])
        if hi >= 6 or n >= 10:
            row["Betrouwbaarheid"] = "goed"
        elif hi >= 3 or n >= 5:
            row["Betrouwbaarheid"] = "bruikbaar"
        elif n >= 1:
            row["Betrouwbaarheid"] = "beperkt"
        else:
            row["Betrouwbaarheid"] = "onvoldoende"
    df = pd.DataFrame(list(base_keys.values()))
    if not df.empty:
        df = df.sort_values(["Team", "Momenten", "High/med-high"], ascending=[True, False, False])
    return df


def generate_team_screening_report(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    video_metadata: Dict[str, Any],
    team_a_name: str,
    team_a_color: str,
    team_b_name: str,
    team_b_color: str,
    lineups: List[Dict[str, str]],
    moments: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    max_players_in_report: int = 28,
) -> Dict[str, Any]:
    prompt = f"""
Maak een professioneel Nederlandstalig TEAM SCREENING-rapport voor Club Brugge op basis van een wedstrijdvideo.
Gebruik uitsluitend de AI-observaties hieronder. Geen externe kennis en geen fictieve conclusies.

Doel van deze module:
- niet: 22 volwaardige scoutingsrapporten;
- wel: korte eerste screenings per speler + shortlist/rewatch-advies.

Teams:
- {team_a_name} ({team_a_color})
- {team_b_name} ({team_b_color})

Videometadata:
{json.dumps(video_metadata, ensure_ascii=False, indent=2)}

Line-ups:
{json.dumps(lineups, ensure_ascii=False, indent=2)}

Samenvatting per speler:
{json.dumps(summary_rows, ensure_ascii=False, indent=2)}

Geselecteerde observaties/momenten:
{json.dumps(moments[:350], ensure_ascii=False, indent=2)}

Schrijf output als geldig JSON-object zonder markdown:
{{
  "report_type": "Team Screening",
  "match_summary": "korte algemene samenvatting van de betrouwbaarheid en wat de scan opleverde",
  "method_note": "korte nuance: dit is een AI-screening, geen volledig scoutingsrapport",
  "team_screenings": [
    {{
      "team": "teamnaam",
      "shirt_number": "4",
      "player_name": "naam",
      "position": "positie",
      "visibility": "goed/bruikbaar/beperkt/onvoldoende",
      "contact_moments": 7,
      "screening_text": "80 tot 130 woorden. Begin positief, eindig met duidelijke beperkingen/werkpunten. Verwerk Club Brugge-principes waar zichtbaar.",
      "general_conclusion": "30 tot 55 woorden met advieswaarde",
      "recommendation": "No follow / Rewatch / Keep monitoring / Actively follow / Target",
      "reliability": "high/medium_high/medium/low"
    }}
  ],
  "shortlist": ["spelers die op basis van deze beperkte scan verder bekeken mogen worden"],
  "rewatch_list": ["spelers met te weinig maar potentieel interessante data"],
  "overall_conclusion": "korte conclusie voor Club Brugge"
}}

Maak maximaal {max_players_in_report} spelers in team_screenings. Als spelers uit de line-up geen momenten hebben: neem ze kort op met visibility='onvoldoende' en recommendation='Rewatch' of 'No follow', zonder prestatieconclusies.
""".strip()
    args: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 7000,
    }
    if reasoning_effort and reasoning_effort != "none":
        args["reasoning"] = {"effort": reasoning_effort}
    response = client.responses.create(**args)
    return extract_json_object(response_output_text(response))


def build_team_screening_pdf(
    video_metadata: Dict[str, Any],
    team_a_name: str,
    team_b_name: str,
    team_summary_df: pd.DataFrame,
    moments: List[Dict[str, Any]],
    final_report: Dict[str, Any],
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.3 * cm, bottomMargin=1.3 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DVSTeamTitle", parent=styles["Title"], textColor=colors.HexColor("#0B2A4A"), fontSize=20, leading=24)
    h2 = ParagraphStyle("DVSTeamH2", parent=styles["Heading2"], textColor=colors.HexColor("#0E5D91"), fontSize=13, leading=17, spaceBefore=9)
    body = ParagraphStyle("DVSTeamBody", parent=styles["BodyText"], fontSize=8.8, leading=11)
    small = ParagraphStyle("DVSTeamSmall", parent=styles["BodyText"], fontSize=7.5, leading=9.5, textColor=colors.HexColor("#444444"))
    story: List[Any] = []
    story.append(pdf_paragraph("Digital Video Scout · Team Screening", title_style))
    story.append(pdf_paragraph(f"{team_a_name} - {team_b_name}", h2))
    story.append(pdf_paragraph(f"Rapportdatum: {datetime.now().strftime('%Y-%m-%d %H:%M')} · {APP_VERSION} · max upload 3 GB", small))
    story.append(Spacer(1, 0.25 * cm))
    story.append(pdf_paragraph(final_report.get("match_summary", ""), body))
    story.append(pdf_paragraph(final_report.get("method_note", ""), small))
    story.append(Spacer(1, 0.25 * cm))

    meta_rows = [
        ["Video", video_metadata.get("filename", "")],
        ["Duur", f"{video_metadata.get('duration_seconds', 0)} sec"],
        ["Frames gescand", str(video_metadata.get("sampled_frames", 0))],
        ["Gevonden momenten", str(len(moments))],
    ]
    meta_table = Table(meta_rows, colWidths=[4.0 * cm, 11.5 * cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0B2A4A")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.35 * cm))

    story.append(pdf_paragraph("Overzicht per speler", h2))
    if not team_summary_df.empty:
        rows = [["Team", "#", "Speler", "Positie", "Momenten", "Betrouwbaarheid"]]
        for _, r in team_summary_df.head(34).iterrows():
            rows.append([str(r.get("Team", ""))[:18], str(r.get("#", "")), str(r.get("Speler", ""))[:24], str(r.get("Positie", ""))[:18], str(r.get("Momenten", "")), str(r.get("Betrouwbaarheid", ""))])
        table = Table(rows, colWidths=[2.7 * cm, 0.9 * cm, 4.1 * cm, 3.0 * cm, 1.7 * cm, 3.1 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2A4A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
            ("FONTSIZE", (0, 0), (-1, -1), 7.0),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
    story.append(Spacer(1, 0.35 * cm))

    story.append(pdf_paragraph("Korte screenings", h2))
    for item in final_report.get("team_screenings", [])[:34]:
        header = f"#{item.get('shirt_number','')} {item.get('player_name','Onbekend')} · {item.get('team','')} · {item.get('position','')} · {item.get('visibility','')}"
        story.append(pdf_paragraph(header, h2))
        story.append(pdf_paragraph(str(item.get("screening_text", "")), body))
        story.append(pdf_paragraph(f"Conclusie: {item.get('general_conclusion','')}", small))
        story.append(pdf_paragraph(f"Advies: {item.get('recommendation','')} · betrouwbaarheid: {item.get('reliability','')}", small))
        story.append(Spacer(1, 0.12 * cm))

    story.append(PageBreak())
    story.append(pdf_paragraph("Momentenlog", title_style))
    rows = [["Tijd", "Team", "#", "Speler", "Actie", "Conf.", "Detail"]]
    for m in moments[:150]:
        rows.append([
            str(m.get("timecode", "")), str(m.get("team", ""))[:14], str(m.get("shirt_number", "")),
            str(m.get("player_name", ""))[:18], str(m.get("action_type", ""))[:16], str(m.get("identity_confidence", "")), str(m.get("detail", ""))[:105],
        ])
    if len(rows) == 1:
        rows.append(["—", "—", "—", "—", "—", "—", "Geen bruikbare momenten gevonden."])
    table = Table(rows, colWidths=[1.5 * cm, 2.2 * cm, 0.8 * cm, 2.9 * cm, 2.4 * cm, 1.6 * cm, 5.0 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2A4A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("FONTSIZE", (0, 0), (-1, -1), 6.6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    doc.build(story)
    return buffer.getvalue()

def mode_defaults(report_type: str) -> Dict[str, Any]:
    """Cost-aware defaults.

    Screening is intentionally cheap. Team Screening is also budget-oriented:
    it scans the match broadly and creates short reports, not full templates.
    Scouting remains the standard individual report.
    """
    if report_type == "Screening":
        return {
            "model": "gpt-5-mini",
            "reasoning": "low",
            "vision_detail": "low",
            "confidence_threshold": "medium",
            "min_observations": 3,
            "force_report": True,
            "smart_crops": False,
            "interval_seconds": 3.0,
            "max_frames": 60,
            "identity_batch_size": 6,
            "action_batch_size": 3,
            "focus_window_seconds": 4.0,
            "focus_interval_seconds": 1.0,
            "max_width": 720,
            "jpeg_quality": 55,
            "auto_refine": False,
        }
    if report_type == "Team Screening":
        return {
            "model": "gpt-5-mini",
            "reasoning": "low",
            "vision_detail": "low",
            "confidence_threshold": "medium",
            "min_observations": 1,
            "force_report": True,
            "smart_crops": False,
            "interval_seconds": 6.0,
            "max_frames": 220,
            "identity_batch_size": 6,
            "action_batch_size": 3,
            "focus_window_seconds": 4.0,
            "focus_interval_seconds": 1.0,
            "max_width": 720,
            "jpeg_quality": 55,
            "auto_refine": False,
        }
    return {
        "model": "gpt-5-mini",
        "reasoning": "medium",
        "vision_detail": "auto",
        "confidence_threshold": "medium",
        "min_observations": 5,
        "force_report": True,
        "smart_crops": True,
        "interval_seconds": 1.5,
        "max_frames": 140,
        "identity_batch_size": 5,
        "action_batch_size": 2,
        "focus_window_seconds": 6.0,
        "focus_interval_seconds": 1.0,
        "max_width": 960,
        "jpeg_quality": 65,
        "auto_refine": True,
    }

def main() -> None:
    set_page_style()

    # Sidebar: connection + compact advanced controls.
    st.sidebar.markdown("## Verbinding")
    secret_key = safe_get_secret("OPENAI_API_KEY")
    key_source = "Streamlit Secrets / environment" if secret_key else "Nog geen key gevonden"
    st.sidebar.caption(f"Key bron: {key_source}")
    manual_key = ""
    if not secret_key:
        manual_key = st.sidebar.text_input("OpenAI API key", type="password", help="Voor online gebruik: zet dit liever in Streamlit Secrets.")
    api_key = secret_key or manual_key.strip()

    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "connection_message" not in st.session_state:
        st.session_state.connection_message = "Nog niet getest."

    connection_model = st.sidebar.text_input("Model voor verbindingstest", value=DEFAULT_MODEL)
    if st.sidebar.button("Verbinding maken", use_container_width=True):
        with st.sidebar.status("Verbinding testen...", expanded=False):
            ok, msg = test_openai_connection(api_key=api_key, model=connection_model, reasoning_effort="low")
        st.session_state.connected = ok
        st.session_state.connection_message = msg

    if st.session_state.connected:
        st.sidebar.success("Verbonden")
    else:
        st.sidebar.warning("Niet verbonden")
    st.sidebar.caption(st.session_state.connection_message)

    hero()
    # 1. Video
    st.subheader("1. Video")
    video_file = None
    video_url = ""
    video_input_mode = st.radio(
        "Videobron",
        ["Upload", "Google Drive / link"],
        horizontal=True,
        help="Gebruik upload voor korte clips. Gebruik Google Drive/link voor grotere wedstrijden of wanneer uploaden blokkeert.",
    )
    if video_input_mode == "Upload":
        video_file = large_file_uploader(
            "Upload wedstrijd of geknipte beelden",
            type=["mp4", "mov", "m4v", "avi", "mkv"],
            help="Uploadlimiet staat op 3000 MB via .streamlit/config.toml én per-widget max_upload_size. Voor Team Screening mag dit een volledige wedstrijd zijn, maar upload/verwerking kan traag zijn.",
        )
        limit_mb = active_upload_limit_mb()
        if limit_mb and limit_mb < UPLOAD_LIMIT_MB:
            st.warning(f"Actieve Streamlit uploadlimiet lijkt {limit_mb} MB. Dan staat .streamlit/config.toml waarschijnlijk niet in de root van je GitHub-repo of de app is nog niet volledig gereboot.")
        else:
            st.caption(f"Uploadlimiet ingesteld op {UPLOAD_LIMIT_MB} MB / 3 GB.")
    else:
        video_url = st.text_input(
            "Google Drive-link of directe videolink",
            value="",
            placeholder="Plak hier je Google Drive share link of een directe .mp4-link",
            help="Zet je Google Drive-bestand op: Iedereen met de link kan bekijken. De app downloadt de video tijdelijk naar de server.",
        )
        st.caption("Tip: Google Drive-link werkt best als het bestand gedeeld is met ‘Iedereen met de link’. Voor zeer grote bestanden blijft verwerking op Streamlit Cloud afhankelijk van geheugen en tijd.")

    # 2. Optie
    st.subheader("2. Optie")
    report_type = st.radio(
        "Wat wil je maken?",
        ["Screening", "Scouting", "Team Screening"],
        horizontal=True,
        help="Team Screening maakt korte screenings per speler. Screening/Scouting focussen op één doelspeler.",
    )
    defaults = mode_defaults(report_type)

    cols = st.columns(3)
    with cols[0]:
        st.markdown("""<div class="dvs-card"><b>Screening</b><br/>Budget · één speler · eerste indruk · samenhangende tekst + algemene conclusie.</div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""<div class="dvs-card"><b>Scouting</b><br/>Standard · één speler · contactmomenten · positie-template + PDF.</div>""", unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""<div class="dvs-card"><b>Team Screening</b><br/>Budget · volledige wedstrijd · korte screening per speler · shortlist.</div>""", unsafe_allow_html=True)

    # Sidebar settings depend on chosen mode.
    with st.sidebar.expander("⚙️ Kosten & analyse-instellingen", expanded=False):
        st.caption("Standaardinstellingen worden automatisch gekozen. Pas enkel aan als je bewust meer/minder kost wil.")
        model = st.text_input("Analysemodel", value=defaults["model"])
        reasoning_effort = st.selectbox("Reasoning", ["low", "medium", "high", "none"], index=["low", "medium", "high", "none"].index(defaults["reasoning"]))
        vision_detail = st.selectbox("Vision detail", ["low", "auto", "high"], index=["low", "auto", "high"].index(defaults["vision_detail"]))
        confidence_threshold = st.selectbox("Min. confidence", ["high", "medium_high", "medium"], index=["high", "medium_high", "medium"].index(defaults["confidence_threshold"]))
        min_observations = st.slider("Min. herkenningsframes", 1, 30, defaults["min_observations"], 1)
        force_report = st.checkbox("Beperkt rapport toelaten", value=defaults["force_report"])
        smart_crops = st.checkbox("Smart crops/zoom meesturen", value=defaults["smart_crops"])
        auto_refine = st.checkbox("Auto Player Lock verfijnen", value=defaults["auto_refine"], help="Voor individuele scouting: gebruikt gevonden frames als extra referentie en scant opnieuw.")
        interval_seconds = st.slider("Frame-interval herkenning", 0.5, 12.0, defaults["interval_seconds"], 0.5)
        max_frames = st.slider("Max frames scannen", 20, 700, defaults["max_frames"], 10)
        identity_batch_size = st.slider("Frames per scanbatch", 1, 10, defaults["identity_batch_size"], 1)
        action_batch_size = st.slider("Contactmomenten per batch", 1, 5, defaults["action_batch_size"], 1)
        focus_window_seconds = st.slider("Focuswindow", 2.0, 20.0, defaults["focus_window_seconds"], 1.0)
        focus_interval_seconds = st.slider("Focus frame-interval", 0.5, 3.0, defaults["focus_interval_seconds"], 0.5)
        max_width = st.select_slider("Max beeldbreedte", options=[480, 640, 720, 960, 1280], value=defaults["max_width"])
        jpeg_quality = st.slider("JPEG kwaliteit", 35, 90, defaults["jpeg_quality"], 5)

    # 3. Input depending on mode
    st.subheader("3. Input")
    team_lineups: List[Dict[str, str]] = []
    team_a_name = ""
    team_a_color = ""
    team_b_name = ""
    team_b_color = ""
    scan_scope = "Beide teams"

    if report_type == "Team Screening":
        st.caption("Vul bij voorkeur de line-ups in. Zonder line-up kan de app alleen werken met teamkleur/rugnummer en krijg je meer 'Onbekend'.")
        tcols = st.columns(3)
        with tcols[0]:
            scan_scope = st.selectbox("Wie screenen?", ["Beide teams", "Team A", "Team B"], index=0)
        with tcols[1]:
            team_a_name = st.text_input("Team A", value="PSG")
            team_a_color = st.text_input("Teamkleur A", value="blauw")
        with tcols[2]:
            team_b_name = st.text_input("Team B", value="Tegenstander")
            team_b_color = st.text_input("Teamkleur B", value="wit")

        lcols = st.columns(2)
        with lcols[0]:
            lineup_a_raw = st.text_area(
                "Line-up Team A",
                value="4 Harryl Mboma - Centrale verdediger\n",
                height=180,
                help="Formaat: rugnummer naam - positie. Eén speler per lijn.",
            )
        with lcols[1]:
            lineup_b_raw = st.text_area(
                "Line-up Team B",
                value="",
                height=180,
                help="Optioneel. Formaat: rugnummer naam - positie. Eén speler per lijn.",
            )
        team_lineups = []
        if scan_scope in ("Beide teams", "Team A"):
            team_lineups.extend(parse_lineup_text(lineup_a_raw, team_a_name, team_a_color))
        if scan_scope in ("Beide teams", "Team B"):
            team_lineups.extend(parse_lineup_text(lineup_b_raw, team_b_name, team_b_color))
        parsed_df = pd.DataFrame(team_lineups)
        if not parsed_df.empty:
            with st.expander("Gelezen spelerslijst controleren", expanded=False):
                st.dataframe(parsed_df, use_container_width=True)
        else:
            st.warning("Nog geen spelers uit de line-up gelezen. De module kan dan wel scannen, maar namen koppelen wordt onbetrouwbaar.")
    else:
        st.caption("Hou dit eenvoudig: naam, teamkleur en rugnummer zijn de basis. Eén player-lock beeld of korte uiterlijke hint is genoeg als het rugnummer moeilijk zichtbaar is.")
        c1, c2, c3 = st.columns([1.2, 1, 0.7])
        with c1:
            player_name = st.text_input("Spelernaam", value="Harryl Mboma")
            team_name = st.text_input("Team", value="PSG")
        with c2:
            team_color = st.text_input("Teamkleur", value="blauw")
            shirt_number = st.text_input("Rugnummer", value="4")
        with c3:
            position = st.selectbox("Positie", [
                "Centrale verdediger", "Rechter centrale verdediger", "Linker centrale verdediger", "Wingback", "Flankverdediger",
                "Controlerende middenvelder", "Centrale middenvelder", "Aanvallende middenvelder", "Winger", "Spits", "Doelman", "Andere"
            ], index=0)

        with st.expander("Optioneel: herkenning verbeteren"):
            reference_uploads = large_file_uploader(
                "Player-lock beeld, optioneel",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                help="Bij voorkeur één duidelijke crop/screenshot van de speler met rugnummer zichtbaar.",
            )
            appearance_hint = st.text_area(
                "Uiterlijke hint, optioneel",
                value="",
                placeholder="Bijv. lange mouwen, gele schoenen, linker/rechter centrale verdediger, opvallend kapsel...",
                height=70,
            )
            position_hint = POSITION_HINTS.get(position, f"Positie: {position}. Gebruik teamkleur, rugnummer en positiecontext om de doelspeler te herkennen.")
            with st.expander("Geavanceerd: alleen als herkenning moeilijk blijft"):
                anchor_timestamps = st.text_area(
                    "Player-lock timestamps, optioneel",
                    value="",
                    placeholder="Bijv. 00:12, 01:24, 02:33",
                    height=70,
                )
                focus_timestamps = st.text_area(
                    "Focus-timestamps, optioneel",
                    value="",
                    placeholder="Bijv. 00:30, 01:15, 03:48",
                    height=70,
                )
                custom_position_hint = st.text_area("Positionele hint aanpassen, optioneel", value=position_hint, height=70)
                if custom_position_hint.strip():
                    position_hint = custom_position_hint.strip()

    # 4. Start
    st.subheader("4. Start")
    st.caption(f"Gekozen modus: **{report_type}** · standaardmodel: **{defaults['model']}** · video max 3 GB.")

    if report_type == "Team Screening":
        has_required_input = (video_file is not None or bool(video_url.strip())) and (bool(team_lineups) or bool(team_a_name.strip()) or bool(team_b_name.strip()))
        start_label = "▶ Start Team Screening"
    else:
        has_required_input = (video_file is not None or bool(video_url.strip())) and bool(player_name.strip()) and bool(shirt_number.strip())
        start_label = f"▶ Start {report_type}"

    can_start = st.session_state.connected and has_required_input
    if not st.session_state.connected:
        st.info("Klik eerst links in de sidebar op ‘Verbinding maken’. De startknop blijft geblokkeerd tot dat lukt.")
    start_button = st.button(start_label, type="primary", disabled=not can_start, use_container_width=True)

    if not start_button:
        return

    try:
        client = make_openai_client(api_key)

        if report_type == "Team Screening":
            with st.status("Wedstrijd voorbereiden en frames extraheren...", expanded=True) as status:
                frames, metadata = extract_frames_team_screening(
                    video_file=video_file,
                    video_url=video_url,
                    interval_seconds=interval_seconds,
                    max_regular_frames=max_frames,
                    max_width=max_width,
                    quality=jpeg_quality,
                )
                st.write(f"{len(frames)} frames geëxtraheerd uit {metadata.get('duration_seconds')} seconden video.")
                st.write(f"{len(team_lineups)} spelers in line-up/spelerslijst.")
                status.update(label="Wedstrijd voorbereid", state="complete")

            st.subheader("Wedstrijdscan")
            batches = split_batches(frames, identity_batch_size)
            team_results: List[Dict[str, Any]] = []
            progress = st.progress(0)
            for i, batch in enumerate(batches, start=1):
                with st.status(f"Scanbatch {i}/{len(batches)} analyseren...", expanded=False):
                    try:
                        result = analyze_team_frame_batch(
                            client=client,
                            model=model,
                            reasoning_effort=reasoning_effort,
                            vision_detail=vision_detail,
                            batch=batch,
                            team_a_name=team_a_name,
                            team_a_color=team_a_color,
                            team_b_name=team_b_name,
                            team_b_color=team_b_color,
                            lineups=team_lineups,
                            scan_scope=scan_scope,
                        )
                    except Exception as exc:
                        result = {"batch_summary": f"Batch mislukt: {type(exc).__name__}: {exc}", "player_moments": [], "error": str(exc)}
                    team_results.append(result)
                progress.progress(i / max(len(batches), 1))

            moments = aggregate_team_moments(team_results)
            summary_df = summarize_team_moments(moments, team_lineups)
            k1, k2, k3 = st.columns(3)
            k1.metric("Frames gescand", len(frames))
            k2.metric("Spelers in lijst", len(team_lineups))
            k3.metric("Gevonden momenten", len(moments))

            st.markdown("### Overzicht per speler")
            if not summary_df.empty:
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.warning("Geen spelersoverzicht opgebouwd.")

            with st.status("Team Screening rapport genereren...", expanded=False):
                final_report = generate_team_screening_report(
                    client=client,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    video_metadata=metadata,
                    team_a_name=team_a_name,
                    team_a_color=team_a_color,
                    team_b_name=team_b_name,
                    team_b_color=team_b_color,
                    lineups=team_lineups,
                    moments=moments,
                    summary_rows=summary_df.to_dict(orient="records") if not summary_df.empty else [],
                )
                pdf_bytes = build_team_screening_pdf(metadata, team_a_name, team_b_name, summary_df, moments, final_report)

            output = {
                "app_version": APP_VERSION,
                "created_at": datetime.now().isoformat(),
                "report_type": report_type,
                "teams": {
                    "team_a": {"name": team_a_name, "color": team_a_color},
                    "team_b": {"name": team_b_name, "color": team_b_color},
                    "scan_scope": scan_scope,
                },
                "lineups": team_lineups,
                "video_metadata": metadata,
                "settings": {
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "vision_detail": vision_detail,
                    "interval_seconds": interval_seconds,
                    "max_frames": max_frames,
                    "batch_size": identity_batch_size,
                    "max_width": max_width,
                    "jpeg_quality": jpeg_quality,
                },
                "team_results": team_results,
                "moments": moments,
                "summary": summary_df.to_dict(orient="records") if not summary_df.empty else [],
                "final_report": final_report,
            }

            st.success("Team Screening klaar")
            st.subheader("Rapport")
            st.write(final_report.get("match_summary", ""))
            st.caption(final_report.get("method_note", ""))

            screenings = final_report.get("team_screenings", []) or []
            if screenings:
                for item in screenings:
                    with st.expander(f"#{item.get('shirt_number','')} {item.get('player_name','Onbekend')} · {item.get('team','')} · {item.get('visibility','')}"):
                        st.write(item.get("screening_text", ""))
                        st.markdown("**Algemene conclusie**")
                        st.write(item.get("general_conclusion", ""))
                        st.caption(f"Advies: {item.get('recommendation','')} · betrouwbaarheid: {item.get('reliability','')}")
            if final_report.get("shortlist"):
                st.markdown("### Shortlist")
                st.write(final_report.get("shortlist"))
            if final_report.get("rewatch_list"):
                st.markdown("### Rewatch")
                st.write(final_report.get("rewatch_list"))
            st.markdown("### Algemene conclusie")
            st.write(final_report.get("overall_conclusion", ""))

            with st.expander("Momentenlog bekijken"):
                if moments:
                    st.dataframe(pd.DataFrame(moments), use_container_width=True)
                else:
                    st.write("Geen momenten gevonden.")

            st.download_button(
                "⬇️ Download Team Screening PDF",
                data=pdf_bytes,
                file_name=f"team_screening_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Download JSON-data",
                data=json.dumps(output, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"team_screening_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )
            return

        # Individual Screening / Scouting branch
        player = PlayerConfig(
            player_name=player_name.strip(),
            team_name=team_name.strip(),
            team_color=team_color.strip(),
            shirt_number=shirt_number.strip(),
            position=position,
            dominant_foot="Onbekend",
            report_context="Club Brugge scouting",
            scouting_template=report_type,
            position_hint=position_hint.strip(),
            appearance_hint=appearance_hint.strip(),
            focus_timestamps=focus_timestamps.strip(),
            anchor_timestamps=anchor_timestamps.strip(),
        )
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
            progress.progress(i / max(len(identity_batches), 1))

        identity_frames = aggregate_identity(identity_results)
        selected_frames, identity_counts = select_frames_for_action(frames, identity_frames, confidence_threshold)

        if auto_refine and selected_frames:
            st.info("Auto Player Lock refinement: gevonden spelerbeelden worden gebruikt als extra referentie voor een tweede scan.")
            auto_refs: List[ReferenceImage] = []
            for fr, row in selected_frames[:6]:
                conf = normalize_confidence(row.get("confidence"))
                if CONFIDENCE_RANK.get(conf, 0) >= CONFIDENCE_RANK.get("medium_high", 3):
                    auto_refs.append(ReferenceImage(label=f"auto_lock_{fr.timecode}_{conf}", jpeg_bytes=fr.jpeg_bytes))
            if auto_refs:
                reference_images = reference_images + auto_refs[:6]
                identity_results_refined: List[Dict[str, Any]] = []
                refine_progress = st.progress(0)
                for i, batch in enumerate(identity_batches, start=1):
                    with st.status(f"Refine identity batch {i}/{len(identity_batches)}...", expanded=False):
                        try:
                            result = analyze_identity_batch(client, model, reasoning_effort, vision_detail, player, batch, reference_images)
                        except Exception as exc:
                            result = {"identity_batch_summary": f"Refine batch mislukt: {type(exc).__name__}: {exc}", "identity_frames": [], "error": str(exc)}
                        identity_results_refined.append(result)
                    refine_progress.progress(i / max(len(identity_batches), 1))
                identity_results = identity_results + identity_results_refined
                identity_frames = aggregate_identity(identity_results)
                selected_frames, identity_counts = select_frames_for_action(frames, identity_frames, confidence_threshold)

        quality = identity_quality_text(identity_counts, min_observations)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Frames gescand", len(frames))
        k2.metric("Target zichtbaar", identity_counts.get("visible_total", 0))
        k3.metric("Contact Lock selected", identity_counts.get("selected_total", 0))
        k4.metric("Identity quality", quality)

        with st.expander("Herkenningslog bekijken"):
            if identity_frames:
                st.dataframe(pd.DataFrame(identity_frames), use_container_width=True)
            else:
                st.write("Geen herkenningsframes gevonden.")

        enough_data = int(identity_counts.get("selected_total", 0) or 0) >= min_observations
        if not enough_data and not force_report:
            st.markdown(
                """
                <div class="dvs-warning">
                    <b>Te weinig betrouwbare spelerherkenning.</b><br/>
                    Er wordt géén volwaardig rapport gemaakt. Voeg één player-lock beeld toe, gebruik een betere clip of kies tijdelijk 'beperkt rapport toelaten'.
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
            context_sec = 1.5 if report_type == "Screening" else 2.5
            contact_windows = build_contact_windows(
                frames=frames,
                selected_frames=selected_frames,
                context_seconds=context_sec,
                cluster_gap_seconds=3.0,
                max_windows=30 if report_type == "Screening" else 80,
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
                report_type=report_type,
            )
            pdf_bytes = build_pdf(player, metadata, identity_counts, identity_frames, observations, counts_df, final_report)

        output = {
            "app_version": APP_VERSION,
            "created_at": datetime.now().isoformat(),
            "report_type": report_type,
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
                "auto_refine": auto_refine,
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
            "contact_window_count": len(contact_windows),
            "action_results": action_results,
            "observations": observations,
            "counts": counts_records,
            "final_report": final_report,
        }

        st.success("Analyse klaar")
        st.subheader("Rapport")
        st.write(final_report.get("identity_summary", ""))
        st.write(final_report.get("executive_summary", ""))
        if report_type == "Screening":
            st.markdown("### Screeningsanalyse")
            st.write(final_report.get("screening_text", ""))
        else:
            overview = final_report.get("overview_table")
            if isinstance(overview, dict) and overview:
                st.markdown("### Overzichtstabel")
                rows = []
                for cat, data in overview.items():
                    if isinstance(data, dict):
                        rows.append({"Categorie": cat, "Score": data.get("score_out_of_10", ""), "Tekst": data.get("text", "")})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.markdown("### Concluderende scoutingsanalyse")
            st.write(final_report.get("scouting_report", ""))
        st.markdown("### Algemene conclusie")
        st.write(final_report.get("general_conclusion", ""))

        if report_type == "Scouting" and not counts_df.empty:
            st.subheader("Datarapport")
            st.dataframe(counts_df, use_container_width=True)
        if observations:
            with st.expander("Actielog bekijken"):
                st.dataframe(pd.DataFrame(observations), use_container_width=True)

        st.download_button(
            "⬇️ Download PDF-rapport",
            data=pdf_bytes,
            file_name=f"{report_type.lower()}_report_{player.player_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download JSON-data",
            data=json.dumps(output, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{report_type.lower()}_data_{player.player_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Analyse mislukt: {type(exc).__name__}: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
