"""
ui.py — Todos os overlays visuais da câmera térmica v3.

Novidades vs v2:
  • draw_stats_panel  — painel min/max/média + follow-the-hot/cold
  • draw_line_profile — gráfico de intensidade ao longo de uma linha
  • draw_magnifier    — lupa 3× no ponto mais quente
  • draw_probe_points — usa ThermalStats; exporta CSV via calib.export_reading()
  • draw_hud          — layout atualizado, mostra R² da calibragem
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from app_state import AppState
from calibration import TwoPointCalibration
from pipeline import COLORMAP_LIST, ThermalStats
from recorder import VideoRecorder


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ── Helpers internos ──────────────────────────────────────────────────────────
def _text_bg(img: np.ndarray, text: str, origin, scale: float,
             color, bg=(0, 0, 0), thickness: int = 1) -> None:
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = origin
    cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 2),
                  bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


# ── 1. Painel de estatísticas ────────────────────────────────────────────────
def draw_stats_panel(
    img: np.ndarray,
    stats: ThermalStats,
    calib: Optional[TwoPointCalibration],
    split_offset: int,
) -> None:
    """Exibe min / max / média e marca visualmente os pontos globais."""
    h, w = img.shape[:2]

    def fmt(intensity: int) -> str:
        val, unit = calib.to_temp(intensity) if calib else (intensity / 255 * 100, "%")
        return f"{val:.1f}{unit}"

    # Marcadores no frame
    hx = stats.hot_x + split_offset
    hy = stats.hot_y
    cx_pt = stats.cold_x + split_offset
    cy_pt = stats.cold_y

    # ● ponto mais quente
    cv2.drawMarker(img, (hx, hy), (0, 0, 255),
                   cv2.MARKER_CROSS, 16, 2)
    _text_bg(img, f"MAX {fmt(stats.max_val)}", (hx + 8, hy - 6),
             0.38, (0, 80, 255))

    # ● ponto mais frio
    cv2.drawMarker(img, (cx_pt, cy_pt), (255, 180, 0),
                   cv2.MARKER_CROSS, 16, 2)
    _text_bg(img, f"MIN {fmt(stats.min_val)}", (cx_pt + 8, cy_pt - 6),
             0.38, (255, 180, 0))

    # Painel no canto superior direito
    panel_w = 160
    px = w - panel_w - 5
    py = 108
    cv2.rectangle(img, (px - 4, py - 4), (w - 2, py + 62), (0, 0, 0), -1)
    cv2.rectangle(img, (px - 4, py - 4), (w - 2, py + 62), (40, 40, 40), 1)

    lines = [
        (f"MAX  {fmt(stats.max_val)}", (0, 80, 255)),
        (f"MED  {fmt(int(stats.mean_val))}", (200, 200, 200)),
        (f"MIN  {fmt(stats.min_val)}", (255, 180, 0)),
    ]
    for i, (txt, color) in enumerate(lines):
        cv2.putText(img, txt, (px, py + 4 + i * 19),
                    FONT, 0.40, color, 1, cv2.LINE_AA)


# ── 2. Pontos de sonda ───────────────────────────────────────────────────────
def draw_probe_points(
    img: np.ndarray,
    enhanced: np.ndarray,
    state: AppState,
    calib: Optional[TwoPointCalibration],
    split_offset: int,
) -> None:
    h, w = enhanced.shape
    readings = []

    for i, p in enumerate(state.probe_points):
        gx = p.x - split_offset
        gy = p.y
        if not (0 <= gx < w and 0 <= gy < h):
            continue

        intensity = int(enhanced[gy, gx])
        temp_val, temp_unit = (calib.to_temp(intensity) if calib
                               else (intensity / 255 * 100, "%"))

        marker_color = cv2.applyColorMap(
            np.array([[intensity]], dtype=np.uint8),
            cv2.COLORMAP_JET)[0][0].tolist()

        cv2.circle(img, (p.x, p.y), 8, marker_color, 2)
        cv2.line(img, (p.x - 5, p.y), (p.x + 5, p.y), marker_color, 1)
        cv2.line(img, (p.x, p.y - 5), (p.x, p.y + 5), marker_color, 1)

        label = f"P{i+1}: {temp_val:.1f}{temp_unit}  ({intensity})"
        lx = min(p.x + 10, img.shape[1] - 100)
        ly = max(p.y - 5, 15)
        _text_bg(img, label, (lx, ly), 0.40, (255, 255, 255))

        # Barra mini
        bx, by = p.x + 10, p.y + 12
        if bx + 50 < img.shape[1] and by + 6 < img.shape[0]:
            cv2.rectangle(img, (bx, by), (bx + 50, by + 5), (60, 60, 60), -1)
            fill = int(intensity / 255 * 50)
            cv2.rectangle(img, (bx, by), (bx + fill, by + 5), marker_color, -1)

        readings.append(dict(px=p.x, py=p.y, intensity=intensity,
                             temp_val=round(temp_val, 2), unit=temp_unit))

    return readings   # retorna para eventual export CSV


# ── 3. Lupa no hotspot ───────────────────────────────────────────────────────
def draw_magnifier(
    img: np.ndarray,
    cx: int,
    cy: int,
    zoom: float = 3.0,
    size: int = 80,
) -> None:
    """Exibe lupa 3× centrada no ponto mais quente (canto inf-direito)."""
    h, w = img.shape[:2]
    half = int(size / zoom / 2)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return

    zoomed = cv2.resize(patch, (size, size), interpolation=cv2.INTER_NEAREST)

    dest_x = w - size - 10
    dest_y = h - size - 10
    img[dest_y:dest_y + size, dest_x:dest_x + size] = zoomed
    cv2.rectangle(img, (dest_x, dest_y),
                  (dest_x + size, dest_y + size), (0, 200, 255), 1)
    cv2.putText(img, f"x{zoom:.0f} HOT", (dest_x + 2, dest_y - 3),
                FONT, 0.30, (0, 200, 255), 1)


# ── 4. Line profile ──────────────────────────────────────────────────────────
def draw_line_profile_chart(
    profile: np.ndarray,
    calib: Optional[TwoPointCalibration],
    colormap_idx: int,
    width: int = 320,
    height: int = 140,
) -> np.ndarray:
    """
    Retorna imagem BGR com o gráfico de intensidade ao longo da linha.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)

    cv2.putText(canvas, "Line Profile", (8, 16),
                FONT, 0.45, (200, 200, 200), 1)

    n = len(profile)
    if n < 2:
        return canvas

    margin = 20
    cw = width - margin * 2
    ch = height - margin * 2 - 10

    # Normaliza para o canvas
    pmin, pmax = int(profile.min()), int(profile.max())
    if pmin == pmax:
        return canvas

    pts = []
    for i, v in enumerate(profile):
        px = margin + int(i / (n - 1) * cw)
        py = margin + ch - int((v - pmin) / (pmax - pmin) * ch)
        pts.append((px, py))

    # Área preenchida
    poly = np.array([[margin, margin + ch]] + pts +
                    [[margin + cw, margin + ch]], dtype=np.int32)
    cv2.fillPoly(canvas, [poly], (30, 30, 60))

    # Linha colorida
    for i in range(len(pts) - 1):
        val = int(profile[i])
        color = cv2.applyColorMap(
            np.array([[val]], dtype=np.uint8),
            COLORMAP_LIST[colormap_idx][1])[0][0].tolist()
        cv2.line(canvas, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)

    # Anotações min/max
    val_min_str, val_max_str = f"{pmin}", f"{pmax}"
    if calib and calib.is_ready():
        t_min, u = calib.to_temp(pmin)
        t_max, _ = calib.to_temp(pmax)
        val_min_str = f"{t_min:.1f}{u}"
        val_max_str = f"{t_max:.1f}{u}"

    cv2.putText(canvas, val_max_str, (margin, margin - 2),
                FONT, 0.30, (200, 200, 200), 1)
    cv2.putText(canvas, val_min_str, (margin, margin + ch + 12),
                FONT, 0.30, (150, 150, 150), 1)

    return canvas


# ── 5. Histograma ────────────────────────────────────────────────────────────
def draw_histogram(
    gray: np.ndarray,
    colormap_idx: int,
    width: int = 320,
    height: int = 200,
) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist, 0, height - 30, cv2.NORM_MINMAX)

    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    cv2.putText(canvas, "Histograma IR", (10, 18),
                FONT, 0.5, (200, 200, 200), 1)

    # Gera todas as 256 cores de uma só vez com LUT vetorizada
    lut_vals = np.arange(256, dtype=np.uint8).reshape(256, 1)
    all_colors = cv2.applyColorMap(lut_vals, COLORMAP_LIST[colormap_idx][1])
    all_colors = all_colors.reshape(256, 3)

    bar_w = max(1, width // 256)
    for i in range(256):
        bar_h_i = int(hist[i, 0])
        if bar_h_i == 0:
            continue
        color = all_colors[i].tolist()
        x0 = i * bar_w
        cv2.rectangle(canvas, (x0, height - bar_h_i - 5),
                      (x0 + bar_w, height - 5), color, -1)

    cv2.putText(canvas, "FRIO", (5, height - 1),
                FONT, 0.35, (150, 150, 255), 1)
    cv2.putText(canvas, "QUENTE", (width - 58, height - 1),
                FONT, 0.35, (255, 150, 150), 1)

    mean_val = int(np.mean(gray))
    mx = mean_val * bar_w
    cv2.line(canvas, (mx, 22), (mx, height - 5), (255, 255, 0), 1)
    cv2.putText(canvas, f"med={mean_val}", (max(0, mx - 20), 20),
                FONT, 0.32, (255, 255, 0), 1)

    return canvas


# ── 6. Overlay de calibragem ─────────────────────────────────────────────────
def draw_calibration_overlay(img: np.ndarray, calib: TwoPointCalibration) -> None:
    if not calib.active:
        return
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 220, 255), 3)
    step = len(calib.ref_pts) + 1
    label = "FRIO" if step == 1 else ("QUENTE" if step == 2 else f"extra {step}")
    msg = (f"CALIBRAGEM — clique ponto {step} ({label})  |  "
           f"K=finalizar ({len(calib.ref_pts)} coletado(s))  |  Shift+K=cancelar")
    (tw, _), _ = cv2.getTextSize(msg, FONT, 0.40, 1)
    cx = max(0, (w - tw) // 2)
    cv2.rectangle(img, (cx - 6, h - 28), (cx + tw + 6, h - 6), (0, 0, 0), -1)
    cv2.putText(img, msg, (cx, h - 11), FONT, 0.40,
                (0, 220, 255), 1, cv2.LINE_AA)


# ── 7. HUD principal ──────────────────────────────────────────────────────────
def draw_hud(
    img: np.ndarray,
    state: AppState,
    calib: TwoPointCalibration,
    recorder: VideoRecorder,
) -> None:
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 104), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    rec_label = " [● REC]" if recorder.active else ""
    title_color = (0, 0, 220) if recorder.active else (0, 200, 255)
    cv2.putText(img, f"CAMERA TERMICA IR v3{rec_label}", (10, 22),
                FONT_BOLD, 0.65, title_color, 1, cv2.LINE_AA)

    cmap_name = COLORMAP_LIST[state.colormap_idx][0]
    line2 = (f"FPS:{state.fps:.1f}  Contraste:{state.alpha:.1f}  "
             f"Colormap:{cmap_name}  Suaviz:{state.smooth_n}f")
    cv2.putText(img, line2, (10, 44), FONT, 0.38, (200, 200, 200), 1)

    flags = []
    if state.split_view:         flags.append("SPLIT")
    if state.hotspot_on:         flags.append("HOTSPOT")
    if state.hist_on:            flags.append("HIST")
    if state.show_line_profile:  flags.append("PROFILE")
    if state.probe_points:       flags.append(f"{len(state.probe_points)} PONTOS")
    if calib.active:
        flags.append(f"CALIB [{len(calib.ref_pts)} coletado(s)]")
    elif calib.is_ready():
        r2 = f" R²={calib.r_squared:.3f}" if calib.r_squared is not None else ""
        flags.append(f"CALIB✓{r2}")
    cv2.putText(img, "Ativos: " + ("  ".join(flags) or "—"),
                (10, 62), FONT, 0.38, (100, 220, 100), 1)

    cv2.putText(img,
                "Q=Sair S=Foto R=Gravar C=Colormap +/-=Contr V=Split H=Hotspot "
                "T=Suaviz G=Hist L=Profile K=Calib(fin.) Shift+K=Reset E=ExportCSV",
                (10, 80), FONT, 0.30, (130, 130, 130), 1)
    cv2.putText(img,
                "Clique-esq=Sonda  Clique-dir=Limpar sondas  "
                "Shift+Clique=Ponto de linha (perfil)",
                (10, 96), FONT, 0.30, (100, 100, 100), 1)

    # Barra de escala de cor — gerada de uma vez com LUT vetorizada
    therm_w = w // 2 if state.split_view else w
    bar_x = therm_w - 30
    bar_h = h - 130
    bar_top = 108
    if bar_h > 0:
        ramp = np.arange(bar_h - 1, -1, -1, dtype=np.float32)
        vals = (ramp / (bar_h - 1) * 255).astype(np.uint8).reshape(-1, 1)
        _, cmap = COLORMAP_LIST[state.colormap_idx]
        bar_colors = cv2.applyColorMap(vals, cmap).reshape(bar_h, 1, 3)
        bar_strip = np.repeat(bar_colors, 15, axis=1)
        img[bar_top:bar_top + bar_h, bar_x:bar_x + 15] = bar_strip
    cv2.putText(img, "Q", (bar_x + 1, bar_top - 4),
                FONT, 0.30, (255, 255, 255), 1)
    cv2.putText(img, "F", (bar_x + 3, bar_top + bar_h + 10),
                FONT, 0.30, (180, 180, 180), 1)
