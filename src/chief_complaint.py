"""Chief complaint normalisation, abbreviation expansion, and categorisation."""

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Abbreviation / synonym dictionary  (lower-case key → canonical form)
# ---------------------------------------------------------------------------
ABBREVIATION_MAP: dict[str, str] = {
    "abd": "abdominal",
    "abd.": "abdominal",
    "ha": "headache",
    "h-headache": "headache",
    "lbp": "low back pain",
    "lbp - low back pain": "low back pain",
    "g/w-general weakness": "general weakness",
    "g/w": "general weakness",
    "f/c-fever/chills": "fever chills",
    "f/c": "fever chills",
    "fb": "foreign body",
    "ptx": "pneumothorax",
    "ptx - pneumothorax": "pneumothorax",
    "loc": "loss of consciousness",
    "loc - loss of consciousness": "loss of consciousness",
    "v - vomiting": "vomiting",
    "nrs": "pain score",
    "ruq": "right upper quadrant",
    "rlq": "right lower quadrant",
    "llq": "left lower quadrant",
    "luq": "left upper quadrant",
    "rt.": "right",
    "rt": "right",
    "lt.": "left",
    "lt": "left",
    "ant.": "anterior",
    "ant": "anterior",
    "sorethroat": "sore throat",
    "nauseavomiting": "nausea vomiting",
    "lac.": "laceration",
    "lac": "laceration",
    "movt.": "movement",
    "movt": "movement",
}

# ---------------------------------------------------------------------------
# 2. Keyword → clinical category mapping
# ---------------------------------------------------------------------------
CATEGORY_RULES: list[tuple[list[str], str]] = [
    # Cardiovascular / chest
    (["chest pain", "chest discomfort", "palpitation", "angina",
      "anterior chest", "chest wall pain", "ischaemic chest"], "cardiovascular"),
    # Respiratory
    (["dyspnea", "shortness of breath", "cough", "pneumothorax",
      "orthopnea", "hyperventilation", "wheezing"], "respiratory"),
    # Gastrointestinal / abdominal
    (["abdominal pain", "epigastric", "vomiting", "nausea", "diarrhea",
      "diarrhoea", "melena", "hematochezia", "constipation",
      "distension", "ascites", "gastric", "periumbilical",
      "flank pain", "upper abdominal", "lower abdominal",
      "right upper quadrant", "right lower quadrant",
      "left lower quadrant", "abdomen"], "gastrointestinal"),
    # Neurological
    (["headache", "seizure", "syncope", "dizziness", "mental change",
      "loss of consciousness", "hemiparesis", "monoparesis",
      "dysarthria", "involuntary movement", "amnesia",
      "convulsion", "post seizure", "behavior change"], "neurological"),
    # Trauma / wound / burn
    (["laceration", "open wound", "fracture", "injury", "burn",
      "trauma", "contusion", "dislocation", "needle stick",
      "foreign body", "pain arm", "pain leg", "finger",
      "hand", "foot", "knee", "hip", "wrist", "ankle",
      "chin pain", "scalp", "eyebrow", "lip laceration"], "trauma"),
    # Infection / fever
    (["fever", "chills", "cellulitis", "abscess", "infection",
      "rash", "urticaria", "eczema", "herpes"], "infection"),
    # Genitourinary / gynecological
    (["vaginal bleeding", "vaginal discharge", "dysuria",
      "hematuria", "urinary", "pregnancy", "uterine",
      "gingival", "spotting"], "genitourinary"),
    # Ophthalmic
    (["ocular pain", "eye pain", "blurred vision", "visual acuity",
      "foreign body in eye", "corneal"], "ophthalmic"),
    # ENT / throat
    (["throat pain", "sore throat", "epistaxis", "ear pain",
      "foreign body in throat", "swelling neck", "epiglottis",
      "pharyngitis", "tonsil"], "ent"),
    # Psychiatric / toxicology
    (["anxiety", "panic", "suicidal", "intoxication", "overdose",
      "drug allergy", "allergy", "anaphylaxis", "alcohol"], "psychiatric_toxicology"),
    # Musculoskeletal / pain
    (["low back pain", "back pain", "myalgia", "pain",
      "swelling", "weakness", "general weakness"], "musculoskeletal"),
]

# UI label → English phrase passed to the model (matches categorise_complaint keywords)
CHIEF_COMPLAINT_DROPDOWN: list[tuple[str, str]] = [
    ("Cardiovascular — chest pain, palpitations", "chest pain"),
    ("Respiratory — shortness of breath, cough", "shortness of breath"),
    ("Gastrointestinal — abdominal pain, nausea", "abdominal pain"),
    ("Neurological — headache, syncope, seizure", "headache"),
    ("Trauma / wound / fracture", "laceration"),
    ("Infection / fever", "fever"),
    ("Genitourinary / gynecology", "dysuria"),
    ("Eye complaint", "ocular pain"),
    ("ENT — throat, ear, nose", "sore throat"),
    ("Psychiatric / intoxication", "anxiety"),
    ("Musculoskeletal — low back / spine pain", "low back pain"),
    ("Other / nonspecific", "routine checkup"),
]


def normalise_complaint(text: str) -> str:
    """Lower-case, expand abbreviations, strip noise."""
    if pd.isna(text):
        return "unknown"
    text = str(text).strip().lower()

    # Unknown / garbage markers
    if text in ("??", "?? ??", "?? ???", "?? ??? ??", ""):
        return "unknown"

    # Expand multi-word abbreviations first (longer keys first)
    for abbr, full in sorted(ABBREVIATION_MAP.items(), key=lambda x: -len(x[0])):
        text = re.sub(r"\b" + re.escape(abbr) + r"\b", full, text)

    # Remove punctuation except hyphens between words
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def categorise_complaint(normalised: str) -> str:
    """Map a normalised complaint string to a clinical category."""
    if normalised == "unknown":
        return "unknown"
    for keywords, category in CATEGORY_RULES:
        for kw in keywords:
            if kw in normalised:
                return category
    return "other"


def add_complaint_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised complaint and category columns to the dataframe."""
    df = df.copy()
    df["cc_normalised"] = df["Chief_complain"].apply(normalise_complaint)
    df["cc_category"] = df["cc_normalised"].apply(categorise_complaint)
    return df
