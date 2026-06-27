"""
recorder.py — Gravação de vídeo em .mp4.
"""

from __future__ import annotations
import time
import cv2


class VideoRecorder:
    def __init__(self, width: int, height: int, fps: float = 20.0) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.active = False
        self.fname = ""
        self._writer: cv2.VideoWriter | None = None

    def start(self) -> None:
        self.fname = f"termica_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self.fname, fourcc, self.fps, (self.width, self.height))
        self.active = True
        print(f"  [REC] Gravando: {self.fname}")

    def write(self, frame: "np.ndarray") -> None:
        if self.active and self._writer:
            import cv2 as _cv2
            out = _cv2.resize(frame, (self.width, self.height))
            self._writer.write(out)

    def stop(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None
        self.active = False
        print(f"  [REC] Salvo: {self.fname}")
