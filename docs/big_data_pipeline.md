# Part 5 — From a Single Camera to an IoT / Big Data Pipeline

This project today produces, per camera:

- ~5 frame-level metric records / second (`frames` table)
- 0..N typed events per second (`events` table)
- 1 row per (class × direction) per minute (`minute_counts` table)

For *one* camera that is small. For a city of *thousands* of cameras —
or the same code re-pointed at retail, ports, tourist sites, parking
lots, beaches, nature reserves — it becomes a true Big Data workload.
Below is the production blueprint we would deploy.

## 1. Reference architecture

```
                     ┌──────────────────────────┐
                     │   N camera stream nodes   │
                     │  (this repo per camera)   │
                     │  - ingest + YOLO + track  │
                     │  - emit JSON events       │
                     └────────────┬──────────────┘
                                  │ MQTT (QoS 1)
                                  ▼
                       ┌──────────────────────┐
                       │  MQTT broker (EMQX,  │
                       │  Mosquitto, HiveMQ)  │
                       └──────────┬───────────┘
                                  │ bridge
                                  ▼
                       ┌──────────────────────┐
                       │   Apache Kafka /     │
                       │   Redpanda topics    │
                       │   - traffic.frames   │
                       │   - traffic.events   │
                       └──────────┬───────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
   ┌───────────────┐    ┌──────────────────┐    ┌─────────────────┐
   │ Spark / Flink │    │  Stream sink ➜   │    │ Real-time alert │
   │ stream jobs   │    │  Parquet on S3   │    │ rules (Flink)   │
   │ (windowed agg)│    │  (data lake)     │    │ → Slack / SMS   │
   └──────┬────────┘    └────────┬─────────┘    └────────┬────────┘
          ▼                      ▼                       ▼
   ┌───────────────┐    ┌──────────────────┐    ┌─────────────────┐
   │ TimescaleDB / │    │ DuckDB / Trino / │    │  PagerDuty,     │
   │ ClickHouse    │    │ Athena ad-hoc    │    │  Ops dashboards │
   └──────┬────────┘    └────────┬─────────┘    └─────────────────┘
          ▼                      ▼
   ┌───────────────┐    ┌──────────────────┐
   │ Grafana /     │    │ Notebooks /      │
   │ Superset      │    │ ML training jobs │
   └───────────────┘    └──────────────────┘
```

## 2. Mapping this codebase to the production stack


| In this repo                           | In production                                 |
| -------------------------------------- | --------------------------------------------- |
| `LiveStream` + `VehicleDetector` (TF)  | One container per camera (k8s `DaemonSet`)    |
| `EventEngine` + `Storage.write_events` | MQTT publisher → `traffic.events` topic       |
| `frames` table writes                  | MQTT publisher → `traffic.frames` topic       |
| `minute_counts` rollup                 | Spark Structured Streaming windowed aggregate |
| SQLite                                 | TimescaleDB (hot, ~30 days) + Parquet on S3   |
| Streamlit dashboard                    | Grafana / Superset, multi-camera              |


The *interfaces* don't change — only the transport (MQTT/Kafka instead
of in-process function calls) and the storage backend (Timescale +
Parquet instead of SQLite). The detector code is the same edge binary.

## 3. Event schema (already implemented)

```jsonc
// LINE_CROSS — one car/truck/bus crossed a configured line
{
  "ts": 1714670400.512,
  "type": "LINE_CROSS",
  "camera_id": "jackson-hole-town-square",
  "class_name": "car",
  "direction": "in",
  "track_id": 1248,
  "payload": {"line": "main_st_NS"}
}

// CONGESTION_HIGH — sustained over threshold for >=10 s
{
  "ts": 1714670503.001,
  "type": "CONGESTION_HIGH",
  "camera_id": "jackson-hole-town-square",
  "payload": {"congestion": 0.74, "n_vehicles": 22, "mean_speed_px": 1.1}
}
```

Adding the `camera_id` field at the publisher and you have a fully
multi-tenant stream — perfectly partitionable by Kafka key.

## 4. Scalability numbers (back-of-the-envelope)

- One camera at stride 5 ≈ **6 frame events/s** + ~0–3 typed events/s.
- 10 000 cameras ≈ **60 k msgs/s on `traffic.frames`**, **<30 k/s on
`traffic.events`**. Comfortable on a 3-broker Kafka cluster.
- Storage: per-camera Parquet ≈ ~~5 MB/day frames, ~1 MB/day events.
10 000 cameras × 365 days ≈ **~~22 TB/year compressed** — typical
data-lake scale.

## 5. Use cases (who buys this data?)

- **City traffic departments** — adaptive signal timing, red-light
duration tuning, congestion forecasting; integrate with SCATS/SCOOT.
- **Public transport ops** — bus-bunching detection, on-time
performance, dynamic dispatch.
- **Urban planners** — modal share over time (cars vs bikes vs buses),
before/after impact studies for new infrastructure.
- **Retail / hospitality** — footfall + vehicle traffic correlation
(you already have an aggregator that handles people too — just
whitelist COCO class 0).
- **Insurance / risk** — chronic-stop locations correlate with crash
risk; underwriting & municipal liability.
- **Environmental / emissions modeling** — feed counts per class into
COPERT / HBEFA emission factors → live `g CO₂/km` heatmaps.
- **Tourism boards** — combine with weather + event calendars to model
visitor flows.
- **Emergency services** — `STOPPED_VEHICLE` events on highways are a
primary signal for incident response.

## 6. ML feedback loop

The Parquet lake becomes labeled training data:

- Sample frames around interesting events → re-label in a tool like CVAT
→ fine-tune the TF Hub SSD MobileNet (or swap in EfficientDet /
CenterNet) on the *site-specific* distribution (lighting, camera
angles, vehicle mix). Export a new SavedModel, push to a TF Serving
cluster or pack into the edge container via OCI registry.
- Aggregated minute counts feed time-series forecasting models
(Prophet, NeuralProphet, Temporal Fusion Transformers) for **demand
prediction** that feeds back into traffic-light control loops.

## 7. Why MQTT *and* Kafka?

- **MQTT** at the edge: lightweight, designed for unreliable
cellular/WAN links between cameras and the central data center, QoS
guarantees, low overhead.
- **Kafka** in the data center: durable replayable log, enables Spark/
Flink stream processing, multiple consumer groups (real-time alerts,
batch ETL, ML training) reading the same stream independently.

A small bridge service (e.g. `mqtt-kafka-connector` or a dozen lines of
Python) ties them together.

## 8. Failure & ops

- `STREAM_DOWN` / `STREAM_UP` events already shipped from the edge feed
an **uptime SLO** dashboard — operators see *which* cameras are dark
in real time without polling them.
- Edge containers are stateless beyond the `traffic.db` checkpoint; on
crash restart they resume — the central pipeline is the source of
truth.

---

The point of this assignment is not the single-camera demo but the
shape it implies: **the same Python module that today writes to SQLite
is the actual edge producer of a city-scale Big Data pipeline.**