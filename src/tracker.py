"""ByteTrack wrapper + LineZone counting.

Each configured line is wrapped so we can emit per-class IN/OUT crossing
events with stable tracker IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import supervision as sv


@dataclass
class CrossingEvent:
    """One vehicle crossed one line."""

    line_name: str
    direction: str            # "in" or "out"
    track_id: int
    class_id: int
    class_name: str


@dataclass
class CountingLine:
    """A single line + its supervision LineZone + per-class tallies."""

    name: str
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    direction_axis: str
    zone: sv.LineZone
    in_by_class: Dict[str, int] = field(default_factory=dict)
    out_by_class: Dict[str, int] = field(default_factory=dict)


class VehicleTracker:
    """ByteTrack + multiple counting lines."""

    def __init__(self, lines_cfg: List[dict], class_name_fn) -> None:
        self.byte_tracker = sv.ByteTrack()
        self._class_name_fn = class_name_fn

        self.lines: List[CountingLine] = []
        for ln in lines_cfg:
            p1 = sv.Point(x=int(ln["p1"][0]), y=int(ln["p1"][1]))
            p2 = sv.Point(x=int(ln["p2"][0]), y=int(ln["p2"][1]))
            self.lines.append(
                CountingLine(
                    name=ln["name"],
                    p1=(p1.x, p1.y),
                    p2=(p2.x, p2.y),
                    direction_axis=ln.get("direction_axis", "y"),
                    zone=sv.LineZone(start=p1, end=p2),
                )
            )

    # ------------------------------------------------------------------ #
    def update(
        self, detections: sv.Detections
    ) -> Tuple[sv.Detections, List[CrossingEvent]]:
        """Run tracker + line crossings; return (tracked_dets, new_events)."""
        tracked = self.byte_tracker.update_with_detections(detections)

        events: List[CrossingEvent] = []
        if len(tracked) == 0:
            return tracked, events

        for line in self.lines:
            crossed_in, crossed_out = line.zone.trigger(detections=tracked)
            for i, (in_, out_) in enumerate(zip(crossed_in, crossed_out)):
                if not (in_ or out_):
                    continue
                class_id = int(tracked.class_id[i]) \
                    if tracked.class_id is not None else -1
                cname = self._class_name_fn(class_id)
                tid = int(tracked.tracker_id[i]) \
                    if tracked.tracker_id is not None else -1
                if in_:
                    line.in_by_class[cname] = line.in_by_class.get(cname, 0) + 1
                    events.append(CrossingEvent(
                        line.name, "in", tid, class_id, cname,
                    ))
                if out_:
                    line.out_by_class[cname] = line.out_by_class.get(cname, 0) + 1
                    events.append(CrossingEvent(
                        line.name, "out", tid, class_id, cname,
                    ))
        return tracked, events

    # ------------------------------------------------------------------ #
    def totals(self) -> Dict[str, Dict[str, int]]:
        """Aggregate counters {line_name: {"in": N, "out": M}}."""
        out: Dict[str, Dict[str, int]] = {}
        for line in self.lines:
            out[line.name] = {
                "in": int(line.zone.in_count),
                "out": int(line.zone.out_count),
            }
        return out
