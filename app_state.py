"""
app_state.py — Estado centralizado da aplicação.
Elimina as variáveis globais e os hacks _ref do v2.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple, Optional
import time
import numpy as np


MAX_PROBE_POINTS = 5


@dataclass
class ProbePoint:
    x: int
    y: int


@dataclass
class AppState:
    # ── câmera ──────────────────────────────────────────────────────────────
    camera_idx: int = 0
    frame_w: int = 640
    frame_h: int = 480

    # ── pipeline ─────────────────────────────────────────────────────────────
    alpha: float = 3.0           # clipLimit do CLAHE
    colormap_idx: int = 0
    smooth_n: int = 5
    hotspot_thresh: int = 220

    # ── UI ───────────────────────────────────────────────────────────────────
    split_view: bool = False
    hotspot_on: bool = False
    hist_on: bool = False
    show_line_profile: bool = False

    # ── pontos de sonda ──────────────────────────────────────────────────────
    probe_points: List[ProbePoint] = field(default_factory=list)

    # ── line profile: dois pontos definidos pelo usuário ─────────────────────
    line_profile_pts: List[Tuple[int, int]] = field(default_factory=list)

    # ── último frame enhanced (uint8 grayscale) ──────────────────────────────
    last_enhanced: Optional[np.ndarray] = None

    # ── FPS ──────────────────────────────────────────────────────────────────
    fps: float = 0.0
    _frame_count: int = field(default=0, repr=False)
    _t_prev: float = field(default_factory=time.time, repr=False)

    def tick(self) -> None:
        """Atualiza o contador de FPS a cada 15 frames."""
        self._frame_count += 1
        if self._frame_count % 15 == 0:
            t_now = time.time()
            self.fps = 15 / (t_now - self._t_prev + 1e-9)
            self._t_prev = t_now

    # ── helpers de sonda ─────────────────────────────────────────────────────
    def add_probe(self, x: int, y: int) -> None:
        # Remove se clicar perto de um ponto existente
        for i, p in enumerate(self.probe_points):
            if abs(p.x - x) < 15 and abs(p.y - y) < 15:
                self.probe_points.pop(i)
                return
        if len(self.probe_points) < MAX_PROBE_POINTS:
            self.probe_points.append(ProbePoint(x, y))

    def clear_probes(self) -> None:
        self.probe_points.clear()

    # ── helpers de line profile ──────────────────────────────────────────────
    def add_line_pt(self, x: int, y: int) -> None:
        if len(self.line_profile_pts) >= 2:
            self.line_profile_pts.clear()
        self.line_profile_pts.append((x, y))

    def clear_line_pts(self) -> None:
        self.line_profile_pts.clear()
