"""Resolve picklist video paths for CV stages."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .stems import id_from_stem, normalize_stem


def resolve_video_path(video_dir: Union[str, Path], stem: str) -> Path:
    """Find a picklist video under ``video_dir`` (common extensions / casing)."""
    video_dir = Path(video_dir)
    normalized = normalize_stem(stem)
    pid = id_from_stem(normalized)
    candidates = [
        video_dir / f"{normalized}.MP4",
        video_dir / f"{normalized}.mp4",
        video_dir / f"{normalized}.MOV",
        video_dir / f"{normalized}.mov",
        video_dir / f"{normalized}.m4v",
        video_dir / f"{normalized}.M4V",
        video_dir / f"picklist_{int(pid)}.MP4",
        video_dir / f"picklist_{int(pid)}.mp4",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No video found for {normalized} under {video_dir}")
