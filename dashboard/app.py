"""Streamlit live dashboard.

Run alongside `python -m src.main`:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ---------------------------------------------------------------------- #
# config + db helpers
# ---------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@st.cache_data(ttl=10)
def _load_db_path() -> str:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return str(ROOT / cfg.get("storage", {}).get(
            "sqlite_path", "data/traffic.db"))
    return str(ROOT / "data" / "traffic.db")


def _connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _read_sql(path: str, sql: str, params=()) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with _connect(path) as con:
            return pd.read_sql_query(sql, con, params=params)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"DB read failed: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------- #
# UI
# ---------------------------------------------------------------------- #
st.set_page_config(
    page_title="Livestream Traffic Analytics",
    page_icon=None,
    layout="wide",
)

st.title("Livestream Traffic Analytics")
st.caption(
    "Live aggregation of vehicle counts, direction of travel, and "
    "congestion index extracted from a public YouTube traffic livestream."
)

with st.sidebar:
    st.header("Settings")
    refresh_seconds = st.slider("Auto-refresh (s)", 2, 30, 5)
    window_minutes = st.slider("Time window (minutes)", 5, 240, 30)
    st.divider()
    db_path = _load_db_path()
    st.code(db_path, language="text")
    if not os.path.exists(db_path):
        st.warning("No database yet — start `python -m src.main` first.")

now = time.time()
window_start = now - window_minutes * 60

# ---------------------------------------------------------------------- #
# pull data
# ---------------------------------------------------------------------- #
frames_df = _read_sql(
    db_path,
    "SELECT ts, n_vehicles, mean_speed_px, density, congestion "
    "FROM frames WHERE ts >= ? ORDER BY ts",
    (window_start,),
)
events_df = _read_sql(
    db_path,
    "SELECT ts, type, class_name, direction, track_id, payload_json "
    "FROM events WHERE ts >= ? ORDER BY ts DESC LIMIT 500",
    (window_start,),
)
minute_df = _read_sql(
    db_path,
    "SELECT minute_ts, class_name, direction, count, avg_congestion "
    "FROM minute_counts WHERE minute_ts >= ? ORDER BY minute_ts",
    (int(window_start),),
)

if not frames_df.empty:
    frames_df["dt"] = pd.to_datetime(frames_df["ts"], unit="s")
if not events_df.empty:
    events_df["dt"] = pd.to_datetime(events_df["ts"], unit="s")
if not minute_df.empty:
    minute_df["dt"] = pd.to_datetime(minute_df["minute_ts"], unit="s")

# ---------------------------------------------------------------------- #
# top KPIs
# ---------------------------------------------------------------------- #
col1, col2, col3, col4 = st.columns(4)

last_cong = float(frames_df["congestion"].iloc[-1]) if not frames_df.empty else 0.0
last_n = int(frames_df["n_vehicles"].iloc[-1]) if not frames_df.empty else 0
last_speed = float(frames_df["mean_speed_px"].iloc[-1]) if not frames_df.empty else 0.0

cross_df = events_df[events_df["type"] == "LINE_CROSS"] if not events_df.empty else pd.DataFrame()
total_crossings = int(len(cross_df))

col1.metric("Vehicles in view (latest)", last_n)
col2.metric("Mean speed (px/frame)", f"{last_speed:.2f}")
col3.metric(f"Crossings (last {window_minutes}m)", total_crossings)
col4.metric("Congestion (latest)", f"{last_cong:.2f}")

# ---------------------------------------------------------------------- #
# congestion gauge
# ---------------------------------------------------------------------- #
gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=round(last_cong, 2),
        number={"valueformat": ".2f"},
        title={"text": "Congestion index (0..1)"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "rgba(255,165,0,0.85)"},
            "steps": [
                {"range": [0, 0.6],  "color": "rgba(0, 200, 0, 0.25)"},
                {"range": [0.6, 0.85], "color": "rgba(255, 165, 0, 0.25)"},
                {"range": [0.85, 1.0], "color": "rgba(220, 20, 60, 0.25)"},
            ],
        },
    )
)
gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))

# ---------------------------------------------------------------------- #
# row: gauge + class pie
# ---------------------------------------------------------------------- #
left, right = st.columns([1, 1])
with left:
    st.plotly_chart(gauge, use_container_width=True)

with right:
    if not cross_df.empty:
        pie_df = (
            cross_df.assign(class_name=cross_df["class_name"].fillna("unknown"))
            .groupby("class_name").size().reset_index(name="count")
        )
        fig = px.pie(
            pie_df, names="class_name", values="count",
            title="Crossings by vehicle class",
        )
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for line-crossing events…")

# ---------------------------------------------------------------------- #
# congestion + count timeseries
# ---------------------------------------------------------------------- #
st.subheader("Time series")

ts_left, ts_right = st.columns(2)
with ts_left:
    if not frames_df.empty:
        fig = px.line(
            frames_df, x="dt", y="congestion",
            title="Congestion index",
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No per-frame data yet.")

with ts_right:
    if not minute_df.empty:
        per_min = (
            minute_df.assign(class_name=minute_df["class_name"].replace("", "all"))
            .groupby(["dt", "direction"])["count"].sum().reset_index()
        )
        fig = px.bar(
            per_min, x="dt", y="count", color="direction",
            title="Crossings per minute (IN/OUT)",
            barmode="group",
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for the first minute rollup…")

# ---------------------------------------------------------------------- #
# events table
# ---------------------------------------------------------------------- #
st.subheader("Recent events")
if not events_df.empty:
    show = events_df[
        ["dt", "type", "class_name", "direction", "track_id", "payload_json"]
    ].head(200)
    st.dataframe(show, use_container_width=True, hide_index=True)
else:
    st.info("No events yet.")

# ---------------------------------------------------------------------- #
# auto refresh
# ---------------------------------------------------------------------- #
st.caption(f"Auto-refreshing every {refresh_seconds}s.")
time.sleep(refresh_seconds)
st.rerun()
