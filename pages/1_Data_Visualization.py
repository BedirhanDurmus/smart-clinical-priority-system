"""
Optional multipage entry: same charts as in-app navigation.
Use `streamlit run app.py` and pick **Data visualization** in the sidebar (recommended).
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.dataviz_render import render_data_visualization

st.set_page_config(page_title="Data visualization", page_icon="📊", layout="wide")
render_data_visualization()
