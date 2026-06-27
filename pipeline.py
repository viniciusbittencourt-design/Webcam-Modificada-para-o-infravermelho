"""
pipeline.py — Pipeline de processamento IR.

Melhorias sobre v2:
  • CLAHE instanciado uma vez (não a cada frame)
  • Estatísticas de temperatura em tempo real (min/max/média/follow-the-hot)
  • Line profile: intensidade ao longo de uma linha definida pelo usuário
  • Zoom (lupa) no ponto clicado mais quente
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app_state import AppState

# ── Colormaps disponíveis ────────────────────────────────────────────────────
COLORMAP_LIST = [
    ("INFERNO", cv2.COLORMAP_INFERNO),
    ("JET",     cv2.COLORMAP_JET),
    ("HOT",     cv2.COLORMAP_HOT),
    ("MAGMA",   cv2.COLORMAP_MAGMA),
    ("PLASMA",  cv2.COLORMAP_PLASMA),
    ("BONE",    cv2.COLORMAP_BONE),
    ("RAINBOW", cv2.COLORMAP_RAINBOW),
    ("OCEAN",   cv2.COLORMAP_OCEAN),
]


@dataclass
class ThermalStats:
    min_val: int = 0
    max_val: int = 0
    mean_val: float = 0.0
    hot_x: int = 0
    hot_y: int = 0
    cold_x: int = 0
    cold_y: int = 0


class IRPipeline:
    """
    Encapsula todo o processamento de cada frame:
      1. Suavização temporal
      2. CLAHE (clipLimit ajustável, instância reutilizada)
      3. Colormap
      4. Detecção de hotspots
      5. Split-view
      6. Estatísticas globais
    """

    def __init__(self, alpha: float = 3.0, tile: int = 8) -> None:
        self._alpha = alpha
        self._tile = tile
        self._clahe = self._make_clahe()
        self._smooth_n: int = 5
        self._smooth_buf: deque = deque(maxlen=5)
        self._smooth_sum: Optional[np.ndarray] = None   # soma acumulada float32

    # ── CLAHE ────────────────────────────────────────────────────────────────
    def _make_clahe(self) -> cv2.CLAHE:
        return cv2.createCLAHE(
            clipLimit=self._alpha,
            tileGridSize=(self._tile, self._tile),
        )

    def set_alpha(self, alpha: float) -> None:
        self._alpha = max(0.5, min(alpha, 10.0))
        self._clahe = self._make_clahe()

    def set_smooth_n(self, n: int) -> None:
        n = max(1, n)
        if n != self._smooth_n:
            self._smooth_n = n
            old = list(self._smooth_buf)[-n:]
            self._smooth_buf = deque(old, maxlen=n)
            self._smooth_sum = (
                np.sum(old, axis=0, dtype=np.float32) if old else None
            )

    @property
    def alpha(self) -> float:
        return self._alpha

    def _smooth(self, gray: np.ndarray) -> np.ndarray:
        """Média temporal via soma incremental — O(1) por frame."""
        f = gray.astype(np.float32)
        if self._smooth_sum is None:
            self._smooth_sum = np.zeros_like(f)

        # Remove o frame mais antigo que vai sair do deque
        if len(self._smooth_buf) == self._smooth_n:
            self._smooth_sum -= self._smooth_buf[0]

        self._smooth_buf.append(f)
        self._smooth_sum += f
        return (self._smooth_sum / len(self._smooth_buf)).astype(np.uint8)

    # ── hotspots ─────────────────────────────────────────────────────────────
    @staticmethod
    def detect_hotspots(
        gray: np.ndarray, threshold: int = 220, min_area: int = 80
    ) -> list:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            max_val = int(gray[mask == 255].max())
            spots.append((cx, cy, int(area), max_val, cnt))
        return spots

    # ── estatísticas globais ─────────────────────────────────────────────────
    @staticmethod
    def compute_stats(enhanced: np.ndarray) -> ThermalStats:
        s = ThermalStats()
        s.min_val = int(enhanced.min())
        s.max_val = int(enhanced.max())
        s.mean_val = float(enhanced.mean())
        hot_idx = np.unravel_index(np.argmax(enhanced), enhanced.shape)
        cold_idx = np.unravel_index(np.argmin(enhanced), enhanced.shape)
        s.hot_y, s.hot_x = int(hot_idx[0]), int(hot_idx[1])
        s.cold_y, s.cold_x = int(cold_idx[0]), int(cold_idx[1])
        return s

    # ── line profile ─────────────────────────────────────────────────────────
    @staticmethod
    def line_profile(
        enhanced: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        n_samples: int = 200,
    ) -> np.ndarray:
        """
        Retorna array 1-D com os valores de intensidade ao longo do segmento p1→p2.
        """
        x0, y0 = p1
        x1, y1 = p2
        length = int(np.hypot(x1 - x0, y1 - y0))
        n = max(2, min(n_samples, length))
        xs = np.linspace(x0, x1, n).astype(int).clip(0, enhanced.shape[1] - 1)
        ys = np.linspace(y0, y1, n).astype(int).clip(0, enhanced.shape[0] - 1)
        return enhanced[ys, xs]

    # ── frame completo ───────────────────────────────────────────────────────
    def process(
        self,
        frame: np.ndarray,
        state: AppState,
    ) -> Tuple[np.ndarray, np.ndarray, int, ThermalStats]:
        """
        Processa um frame BGR.

        Retorna:
          out          — imagem BGR para exibição
          enhanced     — grayscale pós-CLAHE (para leituras de ponto)
          split_offset — deslocamento horizontal do lado térmico (split)
          stats        — estatísticas globais
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smooth = self._smooth(gray)
        enhanced = self._clahe.apply(smooth)

        stats = self.compute_stats(enhanced)

        _, cmap = COLORMAP_LIST[state.colormap_idx]
        thermal = cv2.applyColorMap(enhanced, cmap)

        # Hotspots
        if state.hotspot_on:
            spots = self.detect_hotspots(enhanced, threshold=state.hotspot_thresh)
            for (cx, cy, area, max_val, cnt) in spots:
                cv2.drawContours(thermal, [cnt], -1, (0, 255, 255), 2)
                cv2.putText(thermal, f"HOT {max_val}", (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.circle(thermal, (cx, cy), 4, (0, 255, 255), -1)

        # Follow-the-hot: marcador pulsante no ponto mais quente
        pulse = int(time.time() * 4) % 2 == 0
        radius = 12 if pulse else 9
        cv2.circle(thermal, (stats.hot_x, stats.hot_y), radius,
                   (0, 255, 255), 2)
        cv2.circle(thermal, (stats.cold_x, stats.cold_y), radius,
                   (255, 100, 0), 2)

        # Line profile overlay na imagem térmica
        if state.show_line_profile and len(state.line_profile_pts) == 2:
            p1, p2 = state.line_profile_pts
            cv2.line(thermal, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(thermal, p1, 4, (100, 200, 255), -1)
            cv2.circle(thermal, p2, 4, (255, 200, 100), -1)

        # Split-view
        split_offset = 0
        if state.split_view:
            original_bgr = cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)
            out = np.hstack([original_bgr, thermal])
            split_offset = original_bgr.shape[1]
        else:
            out = thermal

        return out, enhanced, split_offset, stats
