"""YouTube livestream ingestor.

Resolves a watch URL to its underlying HLS manifest with `yt-dlp`, then
yields frames from OpenCV with auto-reconnect and periodic re-resolution
(YouTube live URLs rotate every ~hour).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
import yt_dlp

log = logging.getLogger(__name__)


@dataclass
class StreamFrame:
    """One processed frame plus its metadata."""

    index: int           # monotonic index of frames yielded (post-stride)
    raw_index: int       # index in the underlying capture
    timestamp: float     # wall-clock seconds (time.time())
    image: np.ndarray    # BGR, possibly resized
    width: int
    height: int


def resolve_hls_url(youtube_url: str) -> str:
    """Resolve a YouTube watch URL to a directly-readable HLS .m3u8 URL.

    OpenCV cannot read youtube.com/watch URLs directly; it needs the
    underlying media URL.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # Prefer an HLS variant; fall back to whatever yt-dlp resolves.
        "format": "best[protocol^=m3u8]/best",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)

    if "url" in info and info["url"]:
        return info["url"]

    # Some live streams expose multiple formats; pick the first playable one.
    for fmt in info.get("formats", []):
        if fmt.get("url") and fmt.get("protocol", "").startswith("m3u8"):
            return fmt["url"]

    raise RuntimeError(f"Could not resolve a media URL for {youtube_url!r}")


def _open_capture(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    # Small buffer so we stay close to live, not catching up old frames.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass
    return cap


def _resize_keep_aspect(
    img: np.ndarray, target_width: Optional[int]
) -> np.ndarray:
    if not target_width or img.shape[1] == target_width:
        return img
    h, w = img.shape[:2]
    new_h = int(round(h * (target_width / w)))
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)


class LiveStream:
    """Iterable wrapper: reconnects on failure, re-resolves periodically."""

    def __init__(
        self,
        youtube_url: str,
        frame_stride: int = 5,
        resize_width: Optional[int] = 960,
        reresolve_seconds: int = 1800,
        max_read_failures: int = 30,
        on_stream_down=None,   # optional callback(reason: str)
    ) -> None:
        self.youtube_url = youtube_url
        self.frame_stride = max(1, int(frame_stride))
        self.resize_width = resize_width
        self.reresolve_seconds = reresolve_seconds
        self.max_read_failures = max_read_failures
        self._on_stream_down = on_stream_down

        self._cap: Optional[cv2.VideoCapture] = None
        self._hls_url: Optional[str] = None
        self._last_resolve: float = 0.0
        self._raw_idx: int = 0
        self._yield_idx: int = 0

    # ------------------------------------------------------------------ #
    # connection lifecycle
    # ------------------------------------------------------------------ #
    def _ensure_open(self, force_resolve: bool = False) -> None:
        now = time.time()
        need_resolve = (
            force_resolve
            or self._hls_url is None
            or (now - self._last_resolve) > self.reresolve_seconds
        )
        if need_resolve:
            log.info("Resolving HLS URL for %s", self.youtube_url)
            self._hls_url = resolve_hls_url(self.youtube_url)
            self._last_resolve = now
            log.debug("Resolved HLS URL: %s", self._hls_url)
            if self._cap is not None:
                self._cap.release()
                self._cap = None

        if self._cap is None or not self._cap.isOpened():
            self._cap = _open_capture(self._hls_url)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open stream {self._hls_url!r}")
            log.info("Stream open.")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------ #
    # iteration
    # ------------------------------------------------------------------ #
    def frames(self) -> Iterator[StreamFrame]:
        """Generator of `StreamFrame` items, indefinitely."""
        consecutive_failures = 0
        backoff = 1.0

        while True:
            try:
                self._ensure_open()
            except Exception as exc:  # noqa: BLE001
                log.warning("Open failed: %s — backoff %.1fs", exc, backoff)
                if self._on_stream_down:
                    self._on_stream_down(f"open_failed: {exc}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0  # reset after a successful open

            ok, frame = self._cap.read()
            self._raw_idx += 1

            if not ok or frame is None:
                consecutive_failures += 1
                log.debug(
                    "read() failed (%d/%d)",
                    consecutive_failures, self.max_read_failures,
                )
                if consecutive_failures >= self.max_read_failures:
                    log.warning("Too many read failures, reconnecting.")
                    if self._on_stream_down:
                        self._on_stream_down("read_failures")
                    self.close()
                    consecutive_failures = 0
                    # Force a fresh HLS resolve on next loop.
                    self._hls_url = None
                else:
                    time.sleep(0.05)
                continue

            consecutive_failures = 0

            if self._raw_idx % self.frame_stride != 0:
                continue

            img = _resize_keep_aspect(frame, self.resize_width)
            self._yield_idx += 1
            yield StreamFrame(
                index=self._yield_idx,
                raw_index=self._raw_idx,
                timestamp=time.time(),
                image=img,
                width=img.shape[1],
                height=img.shape[0],
            )

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def grab_one(self) -> Tuple[np.ndarray, int, int]:
        """Block until one frame is read; useful for the calibration tool."""
        for f in self.frames():
            return f.image, f.width, f.height
        raise RuntimeError("Stream produced no frames.")
