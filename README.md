# Livestream Traffic Analytics

End-to-end Python + OpenCV application that turns a public YouTube traffic
livestream into structured big-data: vehicle counts by class, direction of
travel, and a derived congestion index — stored in SQLite + CSV and
visualized in a live Streamlit dashboard.

> Postgraduate Program in IoT & Big Data — *Data Extraction from
> Livestreams* assignment.

---

## What it does

1. **Ingest** — resolves a YouTube live URL to its HLS manifest with
  `yt-dlp` and reads frames with OpenCV (with auto-reconnect and periodic
   re-resolution because YouTube live URLs rotate).
2. **Detect** — runs **TensorFlow Hub SSD MobileNet V2** (COCO-trained)
  on every Nth frame, filtered to the configured COCO classes (person,
   car, truck, bus, motorcycle, bicycle). Swap the URL in `config.yaml`
   for EfficientDet/CenterNet without changing any code.
3. **Track** — `supervision.ByteTrack` assigns stable IDs across frames so
  each object is counted exactly once.
4. **Extract features** — line-crossing counts (IN/OUT per direction),
  vehicle density inside an ROI, mean tracker displacement (speed proxy),
   and a 0..1 **congestion index** = `density / (1 + mean_speed_px)`.
5. **Generate events** — typed events for line crossings, sustained high
  congestion, stopped vehicles, wrong-direction tracks, and stream
   outages.
6. **Store** — SQLite (per-frame metrics, append-only event log,
  minute-aggregated counts) with periodic CSV export.
7. **Visualize** — Streamlit dashboard with live counts, class breakdown,
  per-minute time-series, congestion gauge, and recent events table.
8. **Scale up** — `docs/big_data_pipeline.md` shows how the same producer
  plugs into a real IoT/Big Data pipeline (MQTT → Kafka → Spark/Flink →
   TimescaleDB / Parquet on S3 → Grafana / Superset).

---

## Architecture

```
YouTube Live (HLS)
       │
       ▼
 src/stream.py  ── yt-dlp resolves .m3u8, OpenCV reads frames, auto-reconnect
       │
       ▼
 src/detector.py ── TF Hub SSD MobileNet V2 (COCO), filtered to target classes
       │
       ▼
 src/tracker.py  ── ByteTrack + LineZone (IN/OUT counters)
       │
       ▼
 src/features.py ── density, speed proxy, congestion
       │
       ▼
 src/events.py   ── typed events (LINE_CROSS, CONGESTION_HIGH, …)
       │
       ▼
 src/storage.py  ── SQLite + CSV (APScheduler minute rollups)
       │
       ▼
 dashboard/app.py (Streamlit)        docs/big_data_pipeline.md (Part 5)
```

---

## Installation

Tested on Ubuntu 22.04 / Python 3.10–3.12. Dependencies are managed with
[Poetry](https://python-poetry.org/) (the lock file is committed for
fully reproducible installs).

### 1. System dependencies

OpenCV needs `ffmpeg` to read HLS streams. Poetry itself is available
via `pipx`, your distro, or the official installer:

```bash
sudo apt update
sudo apt install -y ffmpeg python3-venv pipx
pipx install poetry          # or: curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Python environment

```bash
git clone <this-repo-url> livestream-traffic
cd livestream-traffic

# Poetry will create an in-project .venv/ thanks to poetry.toml
poetry install
```

`poetry install` reads `pyproject.toml` + `poetry.lock` and produces an
identical environment everywhere. The first detector run downloads the
TF Hub SavedModel (~70 MB for SSD MobileNet V2) into the OS TF-Hub
cache (typically `/tmp/tfhub_modules/`).

To activate the environment for ad-hoc commands:

```bash
poetry shell                 # spawns a sub-shell with the venv active
# or prefix individual commands:
poetry run python -m src.main --help
```

### 3. (Optional) GPU

CPU is fine for SSD MobileNet V2 at the default `frame_stride: 5`. If
you have CUDA + cuDNN installed system-wide, TensorFlow will pick up
the GPU automatically; leave `model.device: ""` to let TF auto-detect,
or set `"cpu"` to force CPU.

---

## Usage

### Pick / verify the stream

`config.yaml` ships with a public 24/7 traffic livestream. Replace
`stream.url` with any public webcam YouTube URL you want to analyze.

> **Ethics**: only use **public** street/traffic cams. This project
> aggregates vehicle counts only — no faces, no licence plates, no
> per-person tracking.

### Calibrate counting lines (one-time per camera)

The first frame opens; left-click two points to define each counting
line, press `n` for the next line, `s` to save, `q` to abort:

```bash
poetry run python -m src.calibrate --config config.yaml
```

### Run the live extractor

```bash
poetry run python -m src.main --config config.yaml --show
```

- `--show` opens an OpenCV window with bounding boxes, track IDs,
counting lines, live counters, and the congestion index. Drop it for
headless / server runs.
- Stop with `Ctrl+C` (or `q` in the window). Final CSVs are flushed on
exit.

Data lands in:

- `data/traffic.db` — SQLite (`frames`, `events`, `minute_counts`)
- `data/events.csv` — append-only events
- `data/minute_counts.csv` — per-minute aggregates

### Launch the dashboard (in another terminal)

```bash
poetry run streamlit run dashboard/app.py
```

The dashboard reads `data/traffic.db` and refreshes every few seconds.

---

## Repository layout

```
.
├── config.yaml               # all tunables: stream, model, lines, thresholds
├── pyproject.toml            # Poetry: project metadata + dependencies
├── poetry.lock               # Poetry: exact resolved versions (committed)
├── poetry.toml               # Poetry: in-project venv config
├── README.md
├── src/
│   ├── stream.py             # yt-dlp HLS resolve + resilient frame iterator
│   ├── calibrate.py          # interactive line-zone picker
│   ├── detector.py           # TF Hub SSD MobileNet V2 detector
│   ├── tracker.py            # ByteTrack + LineZone IN/OUT counters
│   ├── features.py           # density, speed proxy, congestion index
│   ├── events.py             # typed event engine
│   ├── storage.py            # SQLite schema, writers, CSV export, rollups
│   └── main.py               # orchestrator CLI
├── dashboard/
│   └── app.py                # Streamlit live dashboard
├── docs/
│   └── big_data_pipeline.md  # Part 5: scale-up to IoT / Big Data pipeline
├── presentation/
│   └── script.md             # 2-minute video presentation script
└── data/                     # generated CSV + SQLite (sample committed)
```

---

## Assignment mapping


| Assignment part                     | Where                                                  |
| ----------------------------------- | ------------------------------------------------------ |
| Part 1 — open + display livestream  | `src/stream.py` + `src/main.py --show`                 |
| Part 2 — extract useful information | `src/detector.py`, `src/tracker.py`, `src/features.py` |
| Part 3 — generate events            | `src/events.py`                                        |
| Part 4 — store data                 | `src/storage.py` (SQLite + CSV)                        |
| Part 5 — IoT / Big Data context     | `docs/big_data_pipeline.md`                            |


---

## Troubleshooting

- `**Could not open stream**` — re-check the YouTube URL is *live* (not a
past stream) and that `ffmpeg` is installed (`ffmpeg -version`).
- `**HTTP 403 / 429` from yt-dlp** — upgrade: `pip install -U yt-dlp`.
YouTube changes its endpoints; `yt-dlp` ships frequent fixes.
- **Slow / laggy** — increase `stream.frame_stride`, reduce
`stream.resize_width`, or run on a CUDA GPU (TF will pick it up
automatically when `model.device: ""`).
- **No objects detected** — lower `model.conf`, or pick a busier camera.

## Sample data (committed in `data/`)

The repo ships with a real ~4-minute capture from the default Jackson
Hole livestream so reviewers can immediately inspect the data shape:

- `data/events.csv` — line crossings + a stopped-vehicle event
- `data/minute_counts.csv` — per-minute aggregates by class & direction
- `data/sample_annotated_frame.jpg` — what the `--show` overlay looks
like (bounding boxes, track IDs, counting line, HUD, congestion bar)

Re-running `python -m src.main` overwrites these (they are appended/
upserted into a fresh `data/traffic.db`).

## Recording the screencast

To produce the screencast deliverable:

```bash
# Terminal 1 — live extractor with overlay window
poetry run python -m src.main --config config.yaml --show

# Terminal 2 — dashboard
poetry run streamlit run dashboard/app.py
```

Then record the screen (e.g. with OBS, `recordmydesktop`, or
`ffmpeg -f x11grab`). Show the OpenCV window for ~30 s, switch to the
browser to walk through the dashboard for ~60 s, then narrate the
points from `presentation/script.md`.

## License

MIT — code is for educational use. Respect each livestream provider's
terms of service.# IoT_BigData_Livestream
