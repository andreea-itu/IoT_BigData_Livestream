"""Interactive line-zone calibrator.

Grabs one frame from the live stream, lets the user click two points to
define each counting line, and writes the result back to `config.yaml`.

Controls:
  left-click x2  define one line (P1, P2)
  n              start a new line
  u              undo the last point of the current line
  s              save lines to config.yaml and exit
  q / Esc        quit without saving
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import cv2

from .config import load_config, save_config
from .stream import LiveStream

log = logging.getLogger(__name__)
Point = Tuple[int, int]


class _Calibrator:
    def __init__(self, image, window: str = "Calibrate"):
        self.base = image.copy()
        self.window = window
        self.lines: List[List[Point]] = [[]]   # list of point-lists

    # ------------------------------------------------------------------ #
    def _on_mouse(self, event, x, y, flags, _):  # noqa: D401
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        cur = self.lines[-1]
        if len(cur) < 2:
            cur.append((x, y))

    def _draw(self):
        canvas = self.base.copy()
        for i, line in enumerate(self.lines):
            color = (0, 255, 0) if i < len(self.lines) - 1 else (0, 200, 255)
            for p in line:
                cv2.circle(canvas, p, 5, color, -1)
            if len(line) == 2:
                cv2.line(canvas, line[0], line[1], color, 2)
                mid = ((line[0][0] + line[1][0]) // 2,
                       (line[0][1] + line[1][1]) // 2)
                cv2.putText(canvas, f"L{i}", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(
            canvas,
            "click 2 pts | n=new  u=undo  s=save  q=quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
        return canvas

    # ------------------------------------------------------------------ #
    def run(self) -> List[List[Point]]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)

        while True:
            cv2.imshow(self.window, self._draw())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyWindow(self.window)
                return []
            if key == ord("n"):
                if self.lines[-1]:
                    self.lines.append([])
            elif key == ord("u"):
                if self.lines[-1]:
                    self.lines[-1].pop()
            elif key == ord("s"):
                cv2.destroyWindow(self.window)
                # Drop empty trailing line.
                return [ln for ln in self.lines if len(ln) == 2]


def main(config_path: str = "config.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(config_path)

    log.info("Grabbing one live frame for calibration…")
    stream = LiveStream(
        youtube_url=cfg["stream"]["url"],
        frame_stride=1,
        resize_width=cfg["stream"].get("resize_width", 960),
    )
    img, w, h = stream.grab_one()
    stream.close()
    log.info("Got frame %dx%d.", w, h)

    lines = _Calibrator(img).run()
    if not lines:
        log.warning("No lines saved.")
        return

    new_lines = []
    for i, (p1, p2) in enumerate(lines):
        # Heuristic: roughly horizontal line ⇒ direction axis is "y"
        # (vehicles cross by moving up/down). Vertical line ⇒ "x".
        axis = "y" if abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1]) else "x"
        new_lines.append({
            "name": f"line_{i}",
            "p1": [int(p1[0]), int(p1[1])],
            "p2": [int(p2[0]), int(p2[1])],
            "direction_axis": axis,
        })

    cfg["lines"] = new_lines
    save_config(cfg, config_path)
    log.info("Saved %d line(s) to %s", len(new_lines), config_path)


def cli() -> None:
    p = argparse.ArgumentParser(description="Calibrate counting lines.")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()
