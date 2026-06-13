"""Video open, FPS, 4K -> 1080p downscale, frame-index seek."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width_src: int
    height_src: int
    width_proc: int
    height_proc: int
    scale: float


class VideoReader:
    """
    Read video frames at requested indices, downscaled to target height (default 1080).

    ``scale`` is ``processed_height / source_height`` (uniform if aspect preserved).
    """

    def __init__(
        self,
        path: Union[str, Path],
        target_height: int = 1080,
    ) -> None:
        self.path = Path(path)
        self.target_height = int(target_height)
        self._cap: Optional[cv2.VideoCapture] = None
        self.meta: Optional[VideoMeta] = None

    def open(self) -> VideoMeta:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {self.path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 29.97
            logger.warning("CAP_PROP_FPS invalid for %s; defaulting to %.3f", self.path, fps)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w <= 0 or h <= 0:
            cap.release()
            raise OSError(f"Invalid frame size {w}x{h} for {self.path}")

        scale = self.target_height / float(h)
        w_proc = max(1, int(round(w * scale)))
        h_proc = int(self.target_height)

        self._cap = cap
        self.meta = VideoMeta(
            path=self.path,
            fps=fps,
            frame_count=frame_count,
            width_src=w,
            height_src=h,
            width_proc=w_proc,
            height_proc=h_proc,
            scale=scale,
        )
        logger.info(
            "Opened %s: %dx%d @ %.3f fps, %d frames -> proc %dx%d scale=%.4f",
            self.path.name,
            w,
            h,
            fps,
            frame_count,
            w_proc,
            self.target_height,
            scale,
        )
        return self.meta

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def read_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Return BGR uint8 processed frame, or None if read failed."""
        if self._cap is None or self.meta is None:
            raise RuntimeError("VideoReader.open() first")

        cap = self._cap
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            ok, frame = cap.read()
        except cv2.error as e:
            logger.warning("OpenCV error seeking frame %d in %s: %s", frame_index, self.path, e)
            return None

        if not ok or frame is None:
            logger.debug("Failed to read frame %d from %s", frame_index, self.path)
            return None

        if self.meta.scale != 1.0:
            try:
                frame = cv2.resize(
                    frame,
                    (self.meta.width_proc, self.target_height),
                    interpolation=cv2.INTER_AREA,
                )
            except cv2.error as e:
                logger.warning("Resize failed frame %d: %s", frame_index, e)
                return None

        return frame

    def iter_frames(self, indices: Iterable[int]) -> Generator[Tuple[int, np.ndarray], None, None]:
        for idx in indices:
            img = self.read_frame(idx)
            if img is not None:
                yield idx, img
