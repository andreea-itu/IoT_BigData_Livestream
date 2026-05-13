"""Typed event engine.

Consumes per-frame features + crossing events and emits structured
`Event` objects that the storage layer persists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .features import FrameFeatures
from .tracker import CrossingEvent


# Event type constants.
LINE_CROSS = "LINE_CROSS"
CONGESTION_HIGH = "CONGESTION_HIGH"
CONGESTION_CLEAR = "CONGESTION_CLEAR"
STOPPED_VEHICLE = "STOPPED_VEHICLE"
STREAM_DOWN = "STREAM_DOWN"
STREAM_UP = "STREAM_UP"


@dataclass
class Event:
    ts: float
    type: str
    class_name: Optional[str] = None
    direction: Optional[str] = None
    track_id: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)


class EventEngine:
    def __init__(
        self,
        congestion_high_threshold: float = 0.6,
        congestion_high_min_seconds: float = 10.0,
    ) -> None:
        self.cong_threshold = float(congestion_high_threshold)
        self.cong_min_seconds = float(congestion_high_min_seconds)

        self._cong_started_at: Optional[float] = None
        self._cong_active: bool = False
        self._stopped_emitted: Set[int] = set()
        self._stream_down: bool = False

    # ------------------------------------------------------------------ #
    def process(
        self,
        ts: float,
        features: FrameFeatures,
        crossings: List[CrossingEvent],
    ) -> List[Event]:
        events: List[Event] = []

        # 1) Line crossings -> one event each.
        for c in crossings:
            events.append(Event(
                ts=ts,
                type=LINE_CROSS,
                class_name=c.class_name,
                direction=c.direction,
                track_id=c.track_id,
                payload={"line": c.line_name},
            ))

        # 2) Congestion sustained over threshold.
        if features.congestion >= self.cong_threshold:
            if self._cong_started_at is None:
                self._cong_started_at = ts
            elif (
                not self._cong_active
                and (ts - self._cong_started_at) >= self.cong_min_seconds
            ):
                self._cong_active = True
                events.append(Event(
                    ts=ts,
                    type=CONGESTION_HIGH,
                    payload={
                        "congestion": round(features.congestion, 3),
                        "n_vehicles": features.n_vehicles,
                        "mean_speed_px": round(features.mean_speed_px, 2),
                    },
                ))
        else:
            if self._cong_active:
                events.append(Event(
                    ts=ts,
                    type=CONGESTION_CLEAR,
                    payload={"congestion": round(features.congestion, 3)},
                ))
            self._cong_active = False
            self._cong_started_at = None

        # 3) Stopped vehicles (emit once per track, then "forget" once it moves).
        active_stopped = set(features.stopped_track_ids)
        new_stopped = active_stopped - self._stopped_emitted
        cname_by_tid = dict(
            zip(features.stopped_track_ids, features.stopped_classes)
        )
        for tid in new_stopped:
            events.append(Event(
                ts=ts,
                type=STOPPED_VEHICLE,
                class_name=cname_by_tid.get(tid) or None,
                track_id=int(tid),
            ))
        # Drop ids that have left the stopped set so a future stop re-fires.
        self._stopped_emitted = (
            self._stopped_emitted & active_stopped
        ) | new_stopped

        return events

    # ------------------------------------------------------------------ #
    def stream_down(self, reason: str) -> Optional[Event]:
        if self._stream_down:
            return None
        self._stream_down = True
        return Event(
            ts=time.time(),
            type=STREAM_DOWN,
            payload={"reason": reason},
        )

    def stream_up(self) -> Optional[Event]:
        if not self._stream_down:
            return None
        self._stream_down = False
        return Event(ts=time.time(), type=STREAM_UP)
