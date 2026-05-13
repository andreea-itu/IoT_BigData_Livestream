# 2-Minute Video Presentation — Script

> Target length: ~2 min (≈ 280 words). Speak naturally, hit each beat.

---

**[0:00 – 0:15] Hook — what we built**

> "I built a Python application that turns a public YouTube traffic
> livestream into structured Big Data. It detects every vehicle in
> view, tracks it, counts how many are crossing a chosen line in each
> direction, and computes a live congestion index — all in real time."

**[0:15 – 0:35] Stream + tech stack**

> "The source is a public 24/7 traffic livestream. I use `yt-dlp` to
> resolve the underlying HLS URL, OpenCV to read frames,
> **TensorFlow Hub SSD MobileNet V2** (COCO-trained) for object
> detection (people, cars, trucks, buses, motorcycles, bicycles), and
> **ByteTrack** to keep stable IDs across frames so each object is
> counted exactly once."

**[0:35 – 1:00] What information we extract and *why* it's useful**

> "Per minute, the system produces: vehicle counts by class, IN/OUT
> direction across each counting line, average speed proxy, and a 0..1
> congestion index that combines density and how slowly vehicles move.
>
> This is exactly the data city traffic departments use to *time
> traffic lights*, urban planners use to compare modal share before
> and after a bike lane is built, and emergency services use to spot
> stopped vehicles on highways — without paying for a private sensor
> network."

**[1:00 – 1:25] Events and storage**

> "On top of frame-level data, I emit typed events: line crossings,
> sustained high congestion, stopped vehicles, and stream outages.
> Everything lands in a SQLite database with periodic CSV export, and
> a Streamlit dashboard renders live KPIs, a congestion gauge, the
> per-minute crossings histogram, and a recent-events table."

**[1:25 – 1:55] Big Data context**

> "The single-camera prototype is intentionally the *edge node* of a
> real Big Data pipeline. Replace SQLite with an MQTT publisher; bridge
> MQTT into Kafka; let Spark or Flink do windowed aggregations into
> TimescaleDB and Parquet on S3; then plug Grafana, Superset, and ML
> training jobs on top. With ten thousand cameras that's roughly 60 000
> messages per second and ~22 TB of compressed Parquet per year —
> textbook Big Data, and the *exact same edge code* feeds it."

**[1:55 – 2:00] Outro**

> "Code, install instructions, sample CSVs, and the screencast demo
> are on GitHub. Thanks for watching."

---

## Talking points if asked

- *Why congestion index, not just count?* — Density alone doesn't say
  if traffic is moving. Combining density with mean tracker
  displacement gives an intuitive 0..1 that maps to "free flow" /
  "slow" / "jam".
- *How do you handle YouTube URL rotation?* — `LiveStream` re-resolves
  the HLS manifest every 30 minutes and on read failure, with
  exponential backoff, and emits `STREAM_DOWN`/`STREAM_UP` events for
  uptime monitoring.
- *Privacy?* — Only aggregate vehicle counts. No faces, no plates, no
  per-person tracking. Camera is public street; data products are
  statistical.
