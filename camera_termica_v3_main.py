#!/usr/bin/env python3
"""
camera_termica_v3.py — Câmera Térmica IR  (refatoração completa)

Novidades vs v2
───────────────
Funcionais
  • Estatísticas em tempo real: min / max / média exibidas no HUD
  • Follow-the-hot/cold: marcadores automáticos nos pontos extremos do frame
  • Line profile: gráfico de intensidade ao longo de uma linha desenhada
  • Lupa 3× automática no ponto mais quente
  • Calibragem por N pontos (mínimo 2) com regressão linear e R²
  • Export de leituras de pontos para CSV (tecla E)
  • [NOVO] Assinatura Térmica Dinâmica: separa VIVO de OBJETO pela
    variação temporal de IR pixel a pixel (tecla D)

Qualidade de código
  • Sem variáveis globais — todo estado em AppState
  • CLAHE instanciado uma vez fora do loop (IRPipeline)
  • Código separado em módulos: app_state, calibration, pipeline, recorder, ui
  • TwoPointCalibration sem input() no callback do mouse

Requisitos
──────────
    pip install opencv-python numpy

Uso
───
    python camera_termica_v3.py
    python camera_termica_v3.py --camera 1 --smooth 3
"""
import cv2
import argparse
import sys
import time
from collections import deque
from typing import Optional



import numpy as np

from app_state import AppState
from calibration import TwoPointCalibration
from pipeline import IRPipeline
from recorder import VideoRecorder
from ui import (
    draw_calibration_overlay,
    draw_histogram,
    draw_hud,
    draw_line_profile_chart,
    draw_magnifier,
    draw_probe_points,
    draw_stats_panel,
)

WINDOW_MAIN    = "Camera Termica IR v3"
WINDOW_HIST    = "Histograma IR"
WINDOW_PROFILE = "Line Profile"
WINDOW_DYNPLOT = "Assinatura Termica — Variancia por Ponto"

# ─────────────────────────────────────────────────────────────────────────────
#  MÓDULO DE ASSINATURA TÉRMICA DINÂMICA (autocontido, sem globals)
# ─────────────────────────────────────────────────────────────────────────────

DYN_MODES = [
    "OVERLAY",          # variância magma sobreposta ao rainbow IR
    "DINAMICO PURO",    # só a variância, fundo escuro
    "BINARIO VIVO/MORTO",
    "HEATMAP VAR",
]

_DYN_PROBE_COLORS = [
    (0, 200, 255), (0, 255, 100), (255, 180, 0),
    (200, 0, 255), (255, 80, 80), (80, 255, 255),
]


class ThermalBuffer:
    """
    Buffer circular de N frames float32.
    Usa somas incrementais — O(1) por frame, sem np.stack() nem np.std().
    Var = E[X²] - E[X]²  (identidade de König-Huygens)
    """

    def __init__(self, n_frames: int = 60):
        self.n     = n_frames
        self.buf: deque = deque(maxlen=n_frames)
        self._sum:  Optional[np.ndarray] = None
        self._sum2: Optional[np.ndarray] = None

    def push(self, gray: np.ndarray) -> None:
        f = gray.astype(np.float32)
        if self._sum is None:
            self._sum  = np.zeros_like(f)
            self._sum2 = np.zeros_like(f)
        if len(self.buf) == self.n:
            old = self.buf[0]
            self._sum  -= old
            self._sum2 -= old * old
        self.buf.append(f)
        self._sum  += f
        self._sum2 += f * f

    def ready(self) -> bool:
        return len(self.buf) >= max(10, self.n // 3)

    def temporal_std(self) -> np.ndarray:
        n    = len(self.buf)
        mean = self._sum / n
        var  = np.maximum(self._sum2 / n - mean * mean, 0.0)
        return np.sqrt(var)

    def temporal_mean(self) -> np.ndarray:
        return self._sum / len(self.buf)

    def reset(self) -> None:
        self.buf.clear()
        self._sum  = None
        self._sum2 = None

    def resize(self, n: int) -> None:
        self.n  = n
        kept    = list(self.buf)[-n:]
        self.buf = deque(maxlen=n)
        self._sum  = None
        self._sum2 = None
        for f in kept:
            self.push(f.astype(np.uint8))

    def fill_ratio(self) -> float:
        return len(self.buf) / self.n


class DynamicSignatureState:
    """
    Encapsula TODO o estado da assinatura dinâmica.
    Instanciado uma vez em main() e passado por referência.
    """

    def __init__(self, n_frames: int = 60, thresh: float = 4.0):
        self.buffer          = ThermalBuffer(n_frames)
        self.thresh          = thresh
        self.mode_idx        = 0          # índice em DYN_MODES
        self.plot_on         = False
        self.active          = False      # D liga/desliga o modo inteiro
        # histórico de variância por ponto de sonda (compartilha os pontos do AppState)
        self.var_history: dict[int, deque] = {
            i: deque(maxlen=300) for i in range(6)
        }

    def reset_buffer(self) -> None:
        self.buffer.reset()
        for v in self.var_history.values():
            v.clear()

    def next_mode(self) -> None:
        self.mode_idx = (self.mode_idx + 1) % len(DYN_MODES)

    def cycle_buffer(self) -> None:
        new_n = self.buffer.n % 120 + 15   # cicla 15 → 120
        self.buffer.resize(new_n)


def _build_dynamic_image(
    std_map: np.ndarray,
    thresh: float,
    mode: int,
    thermal_bg: np.ndarray,
) -> np.ndarray:
    """Converte o mapa de desvio-padrão para a visualização escolhida."""
    std_norm = np.clip(std_map / (thresh * 3) * 255, 0, 255).astype(np.uint8)

    if mode == 0:   # OVERLAY — rainbow IR + magma onde há variância
        base       = thermal_bg.copy()
        dyn        = cv2.applyColorMap(std_norm, cv2.COLORMAP_MAGMA)
        alive_mask = (std_map > thresh * 0.5).astype(np.float32)
        alive_mask = cv2.GaussianBlur(alive_mask, (21, 21), 0)
        alive_3    = np.stack([alive_mask] * 3, axis=2)
        return (base * (1 - alive_3 * 0.6) + dyn * alive_3 * 0.6).astype(np.uint8)

    elif mode == 1:  # DINÂMICO PURO — magma, fundo escuro para objetos
        out = cv2.applyColorMap(std_norm, cv2.COLORMAP_MAGMA)
        out[(std_map < thresh * 0.5)] = (10, 10, 10)
        return out

    elif mode == 2:  # BINÁRIO VIVO / OBJETO
        binary = (std_map > thresh).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
        binary = cv2.GaussianBlur(binary, (11, 11), 0)
        out    = np.zeros((*binary.shape, 3), dtype=np.uint8)
        out[binary >  127] = (0, 200, 0)
        out[binary <= 127] = (30, 0, 0)
        _, bw = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:
                continue
            cv2.drawContours(out, [cnt], -1, (0, 255, 100), 2)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(out, "VIVO", (cx - 18, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 2)
        return out

    else:            # HEATMAP VAR — JET puro da variância
        return cv2.applyColorMap(std_norm, cv2.COLORMAP_JET)


def _draw_dyn_probe_points(
    img: np.ndarray,
    std_map: np.ndarray,
    mean_map: np.ndarray,
    probes: list,
    dyn: "DynamicSignatureState",
) -> None:
    """Desenha sondas com classificação VIVO / OBJETO sobre o frame dinâmico."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = std_map.shape
    for i, (px, py) in enumerate(probes):
        if not (0 <= px < w and 0 <= py < h):
            continue
        std_val  = float(std_map[py, px])
        mean_val = float(mean_map[py, px])
        alive    = std_val > dyn.thresh
        tag      = "VIVO" if alive else "OBJETO"
        color    = (0, 255, 100) if alive else (80, 80, 255)
        pt_color = _DYN_PROBE_COLORS[i % len(_DYN_PROBE_COLORS)]

        dyn.var_history[i].append(std_val)

        cv2.circle(img, (px, py), 10, pt_color, 2)
        cv2.circle(img, (px, py), 2,  (255, 255, 255), -1)

        label = f"P{i+1} [{tag}] std:{std_val:.1f} ir:{mean_val:.0f}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.40, 1)
        lx = min(px + 13, img.shape[1] - tw - 5)
        ly = max(py - 10, th + 8)
        cv2.rectangle(img, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 3),
                      (0, 0, 0), -1)
        cv2.putText(img, label, (lx, ly), font, 0.40, color, 1, cv2.LINE_AA)

        # mini barra de "vitalidade"
        bx, by = px + 13, py + 14
        if bx + 60 < img.shape[1] and by + 6 < img.shape[0]:
            fill = int(min(std_val / (dyn.thresh * 2), 1.0) * 60)
            cv2.rectangle(img, (bx, by), (bx + 60, by + 5), (40, 40, 40), -1)
            cv2.rectangle(img, (bx, by), (bx + fill, by + 5), color, -1)
            tx = bx + 30
            cv2.line(img, (tx, by - 1), (tx, by + 6), (255, 255, 0), 1)


def _draw_dyn_scale_bar(img: np.ndarray, thresh: float) -> None:
    h   = img.shape[0]
    bx  = img.shape[1] - 45
    bt  = 115
    bh  = h - 130
    for i in range(bh):
        val = int(255 * (1 - i / bh))
        c   = cv2.applyColorMap(np.array([[val]], dtype=np.uint8),
                                cv2.COLORMAP_MAGMA)[0][0].tolist()
        cv2.rectangle(img, (bx, bt + i), (bx + 14, bt + i + 1), c, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Alta", (bx - 2, bt - 4),      font, 0.28, (200, 200, 200), 1)
    cv2.putText(img, "Var",  (bx - 2, bt + 10),     font, 0.28, (200, 200, 200), 1)
    cv2.putText(img, "Baixa",(bx - 4, bt + bh + 10),font, 0.28, (150, 150, 150), 1)

    thresh_y = bt + int(bh * (1 - thresh / 30.0))
    thresh_y = max(bt + 2, min(bt + bh - 2, thresh_y))
    cv2.line(img, (bx - 6, thresh_y), (bx + 16, thresh_y), (255, 255, 0), 2)
    cv2.putText(img, f"lim={thresh:.1f}", (bx - 32, thresh_y - 3),
                font, 0.27, (255, 255, 0), 1)


def _draw_dyn_hud(
    img: np.ndarray,
    fps: float,
    dyn: "DynamicSignatureState",
) -> None:
    """HUD sobreposto no modo assinatura dinâmica."""
    h, w = img.shape[:2]
    ov   = img.copy()
    cv2.rectangle(ov, (0, 0), (w, 115), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.50, img, 0.50, 0, img)

    font    = cv2.FONT_HERSHEY_SIMPLEX
    buf     = dyn.buffer
    pct     = int(buf.fill_ratio() * 100)
    bar_w   = int((w - 200) * buf.fill_ratio())
    ready   = buf.ready()

    # barra de coleta
    cv2.rectangle(img, (10, 95), (10 + w - 200, 107), (40, 40, 40), -1)
    cv2.rectangle(img, (10, 95), (10 + bar_w,   107),
                  (0, 200, 100) if ready else (0, 150, 200), -1)
    status = "ANALISANDO" if ready else f"COLETANDO {pct}%"

    cv2.putText(img, "[ ASSINATURA TERMICA DINAMICA ]", (10, 20),
                font, 0.58, (0, 220, 255), 2)
    cv2.putText(img,
                f"FPS:{fps:.1f}  Limiar:{dyn.thresh:.1f}  "
                f"Buffer:{buf.n}f  Modo:{DYN_MODES[dyn.mode_idx]}  {status}",
                (10, 42), font, 0.36, (200, 200, 200), 1)
    cv2.putText(img,
                "D=Sair modo  N=Modo viz  +/-=Limiar  B=Buffer  "
                "ESPACO=Reset  G=Grafico variancia",
                (10, 62), font, 0.30, (120, 120, 120), 1)
    cv2.putText(img,
                "Clique=Ponto sonda [VIVO=alta var | OBJETO=baixa var]   "
                "Clique-dir=Limpar",
                (10, 80), font, 0.29, (100, 100, 100), 1)


def _draw_variance_plot(
    dyn: "DynamicSignatureState",
    probes: list,
    width: int = 500,
    height: int = 260,
) -> np.ndarray:
    canvas    = np.full((height, width, 3), (15, 15, 15), dtype=np.uint8)
    font      = cv2.FONT_HERSHEY_SIMPLEX
    plot_h    = height - 50
    plot_y    = 25
    max_var   = 30.0

    cv2.putText(canvas, "Variancia Temporal (desvio padrao por ponto)",
                (8, 16), font, 0.38, (180, 180, 180), 1)

    # grade
    for y_tick in range(0, plot_h, plot_h // 5):
        cv2.line(canvas, (40, plot_y + y_tick), (width - 10, plot_y + y_tick),
                 (40, 40, 40), 1)

    for idx, hist in dyn.var_history.items():
        if len(hist) < 2 or idx >= len(probes):
            continue
        vals = list(hist)
        pts  = [
            (40 + int(j / max(len(vals) - 1, 1) * (width - 50)),
             plot_y + plot_h - int(min(v, max_var) / max_var * plot_h))
            for j, v in enumerate(vals)
        ]
        color = _DYN_PROBE_COLORS[idx % len(_DYN_PROBE_COLORS)]
        for j in range(len(pts) - 1):
            cv2.line(canvas, pts[j], pts[j + 1], color, 1)
        cv2.putText(canvas, f"P{idx+1}:{vals[-1]:.1f}",
                    (width - 78, plot_y + 14 + idx * 16),
                    font, 0.32, color, 1)

    # eixo Y
    cv2.putText(canvas, "0",            (28, plot_y + plot_h),
                font, 0.28, (120, 120, 120), 1)
    cv2.putText(canvas, f"{max_var:.0f}", (22, plot_y + 6),
                font, 0.28, (120, 120, 120), 1)
    cv2.putText(canvas, "std",          (5, plot_y + plot_h // 2),
                font, 0.28, (120, 120, 120), 1)

    # linha de limiar
    thresh_y = plot_y + plot_h - int(
        min(dyn.thresh, max_var) / max_var * plot_h)
    cv2.line(canvas, (40, thresh_y), (width - 10, thresh_y), (255, 255, 0), 1)
    cv2.putText(canvas, f"limiar={dyn.thresh:.1f}",
                (width - 100, thresh_y - 3),
                font, 0.28, (255, 255, 0), 1)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  ARGUMENTOS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Câmera térmica IR com análise avançada (v3)")
    p.add_argument("--camera",          type=int,   default=0)
    p.add_argument("--smooth",          type=int,   default=5)
    p.add_argument("--hotspot-thresh",  type=int,   default=220)
    p.add_argument("--dyn-buffer",      type=int,   default=60,
                   help="Frames no buffer da assinatura dinâmica (padrão 60)")
    p.add_argument("--dyn-thresh",      type=float, default=4.0,
                   help="Limiar de std para classificar como VIVO (padrão 4.0)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  MOUSE CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def make_mouse_callback(state: AppState, calib: TwoPointCalibration):
    def callback(event, x, y, flags, param):
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)

        if event == cv2.EVENT_LBUTTONDOWN and calib.active:
            if calib.pending_intensity is None and state.last_enhanced is not None:
                enhanced = state.last_enhanced
                gx = x - (state.frame_w if state.split_view else 0)
                gy = y
                gx = max(0, min(gx, enhanced.shape[1] - 1))
                gy = max(0, min(gy, enhanced.shape[0] - 1))
                calib.add_point(int(enhanced[gy, gx]))
            return

        if event == cv2.EVENT_LBUTTONDOWN and shift:
            state.add_line_pt(x, y)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            state.add_probe(x, y)
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            state.clear_probes()

    return callback


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"\n[ERRO] Não foi possível abrir a câmera {args.camera}.")
        print("  → Tente: python camera_termica_v3.py --camera 1")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  10_000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10_000)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  [CAM] Resolução negociada: {W}x{H}")

    # ── instâncias ────────────────────────────────────────────────────────────
    state    = AppState(
        camera_idx=args.camera, frame_w=W, frame_h=H,
        smooth_n=args.smooth, hotspot_thresh=args.hotspot_thresh,
    )
    pipeline = IRPipeline(alpha=state.alpha)
    pipeline.set_smooth_n(state.smooth_n)
    calib    = TwoPointCalibration()
    recorder = VideoRecorder(W, H)
    dyn      = DynamicSignatureState(
        n_frames=args.dyn_buffer,
        thresh=args.dyn_thresh,
    )

    # ── janela ────────────────────────────────────────────────────────────────
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, W, H)
    cv2.setMouseCallback(WINDOW_MAIN, make_mouse_callback(state, calib))

    print("\n=== CÂMERA TÉRMICA IR v3 ===")
    print("  Q / ESC  — Sair")
    print("  S        — Salvar foto PNG")
    print("  R        — Iniciar/parar gravação")
    print("  C        — Trocar colormap")
    print("  +/-      — Contraste CLAHE")
    print("  V        — Split-view")
    print("  H        — Hotspot detector")
    print("  T        — Suavização temporal (ciclo 1-15)")
    print("  G        — Histograma ao vivo")
    print("  L        — Line profile")
    print("  K        — Calibragem / finalizar coleta")
    print("  Shift+K  — Resetar calibragem")
    print("  E        — Exportar CSV")
    print("  ── ASSINATURA DINÂMICA ──────────────────")
    print("  D        — Ativar/desativar modo assinatura dinâmica")
    print("    (dentro do modo D)")
    print("    N      — Trocar visualização (overlay/puro/binário/heatmap)")
    print("    +/-    — Ajustar limiar VIVO/OBJETO")
    print("    B      — Ciclar tamanho do buffer temporal")
    print("    ESPAÇO — Resetar buffer")
    print("    G      — Gráfico de variância por ponto\n")

    fps         = 0.0
    t_prev      = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        if frame_count % 15 == 0:
            t_now  = time.time()
            fps    = 15 / (t_now - t_prev + 1e-9)
            t_prev = t_now

        state.tick()

        # ── pipeline IR (sempre roda para manter suavização e buffer dyn) ────
        out, enhanced, split_offset, stats = pipeline.process(frame, state)
        state.last_enhanced = enhanced

        # alimenta o buffer dinâmico independentemente do modo ativo
        dyn.buffer.push(enhanced)

        # ── modo ASSINATURA DINÂMICA ─────────────────────────────────────────
        if dyn.active:
            thermal_bg = cv2.applyColorMap(enhanced, cv2.COLORMAP_RAINBOW)

            if dyn.buffer.ready():
                std_map  = dyn.buffer.temporal_std()
                mean_map = dyn.buffer.temporal_mean()
                out      = _build_dynamic_image(
                    std_map, dyn.thresh, dyn.mode_idx, thermal_bg)
                _probes_xy = [(p.x, p.y) for p in state.probe_points]
                _draw_dyn_probe_points(
                    out, std_map, mean_map, _probes_xy, dyn)
            else:
                # ainda coletando — mostra rainbow com aviso
                out = thermal_bg.copy()
                pct = int(dyn.buffer.fill_ratio() * 100)
                cv2.putText(out,
                            f"Coletando dados temporais... {pct}%",
                            (W // 2 - 180, H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                std_map  = np.zeros((H, W), dtype=np.float32)
                mean_map = enhanced.astype(np.float32)

            _draw_dyn_scale_bar(out, dyn.thresh)
            _draw_dyn_hud(out, fps, dyn)

            # gráfico de variância em janela separada
            _probes_xy = [(p.x, p.y) for p in state.probe_points]
            if dyn.plot_on and _probes_xy:
                cv2.imshow(WINDOW_DYNPLOT,
                           _draw_variance_plot(dyn, _probes_xy))
            else:
                try:
                    cv2.destroyWindow(WINDOW_DYNPLOT)
                except Exception:
                    pass

        # ── modo NORMAL ──────────────────────────────────────────────────────
        else:
            try:
                cv2.destroyWindow(WINDOW_DYNPLOT)
            except Exception:
                pass

            draw_stats_panel(out, stats, calib, split_offset)
            draw_probe_points(out, enhanced, state, calib, split_offset)
            draw_magnifier(out, stats.hot_x + split_offset, stats.hot_y)
            draw_calibration_overlay(out, calib)
            draw_hud(out, state, calib, recorder)

            if calib.active and calib.pending_intensity is not None:
                calib.consume_pending(out)

            if state.hist_on:
                hist_img = draw_histogram(enhanced, state.colormap_idx)
                cv2.imshow(WINDOW_HIST, hist_img)
            else:
                try:
                    cv2.destroyWindow(WINDOW_HIST)
                except Exception:
                    pass

            if state.show_line_profile and len(state.line_profile_pts) == 2:
                profile     = pipeline.line_profile(
                    enhanced,
                    state.line_profile_pts[0],
                    state.line_profile_pts[1],
                )
                profile_img = draw_line_profile_chart(
                    profile, calib, state.colormap_idx)
                cv2.imshow(WINDOW_PROFILE, profile_img)
            else:
                try:
                    cv2.destroyWindow(WINDOW_PROFILE)
                except Exception:
                    pass

        # ── gravação (funciona em qualquer modo) ─────────────────────────────
        if recorder.active:
            recorder.write(out)
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(out, (0, 0),
                              (out.shape[1] - 1, out.shape[0] - 1),
                              (0, 0, 200), 4)

        cv2.imshow(WINDOW_MAIN, out)

        # ── teclado ───────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        # ── sair ──────────────────────────────────────────────────────────────
        if key in (ord('q'), ord('Q'), 27):
            if recorder.active:
                recorder.stop()
            break

        # ── teclas COMPARTILHADAS (funcionam em ambos os modos) ───────────────
        elif key in (ord('s'), ord('S')):
            fname = f"termica_{int(time.time())}.png"
            print(fname)
            cv2.imwrite(fname, out)
            print(f"  [FOTO] {fname}")

        elif key in (ord('r'), ord('R')):
            if recorder.active:
                recorder.stop()
            else:
                recorder.start()

        # ── tecla D — alternar modo assinatura dinâmica ───────────────────────
        elif key in (ord('d'), ord('D')):
            dyn.active = not dyn.active
            print(f"  Assinatura Dinamica: {'ON' if dyn.active else 'OFF'}")
            if dyn.active:
                print("    → N=modo  +/-=limiar  B=buffer  ESPAÇO=reset  G=grafico")

        # ── teclas EXCLUSIVAS do modo dinâmico ────────────────────────────────
        elif dyn.active:
            if key in (ord('n'), ord('N')):
                dyn.next_mode()
                print(f"  Modo: {DYN_MODES[dyn.mode_idx]}")

            elif key in (ord('+'), ord('=')):
                dyn.thresh = round(dyn.thresh + 0.5, 1)
                print(f"  Limiar: {dyn.thresh}")

            elif key in (ord('-'), ord('_')):
                dyn.thresh = max(0.5, round(dyn.thresh - 0.5, 1))
                print(f"  Limiar: {dyn.thresh}")

            elif key in (ord('b'), ord('B')):
                dyn.cycle_buffer()
                print(f"  Buffer: {dyn.buffer.n} frames "
                      f"(~{dyn.buffer.n / 20:.1f}s a 20fps)")

            elif key == ord(' '):
                dyn.reset_buffer()
                print("  [RESET] Buffer dinamico limpo.")

            elif key in (ord('g'), ord('G')):
                dyn.plot_on = not dyn.plot_on

        # ── teclas EXCLUSIVAS do modo normal ──────────────────────────────────
        else:
            if key in (ord('c'), ord('C')):
                from pipeline import COLORMAP_LIST
                state.colormap_idx = (state.colormap_idx + 1) % len(COLORMAP_LIST)
                print(f"  Colormap: {COLORMAP_LIST[state.colormap_idx][0]}")

            elif key in (ord('+'), ord('=')):
                pipeline.set_alpha(pipeline.alpha + 0.5)
                state.alpha = pipeline.alpha

            elif key in (ord('-'), ord('_')):
                pipeline.set_alpha(pipeline.alpha - 0.5)
                state.alpha = pipeline.alpha

            elif key in (ord('v'), ord('V')):
                state.split_view = not state.split_view

            elif key in (ord('h'), ord('H')):
                state.hotspot_on = not state.hotspot_on
                print(f"  Hotspot: {'ON' if state.hotspot_on else 'OFF'}")

            elif key in (ord('t'), ord('T')):
                state.smooth_n = state.smooth_n % 15 + 1
                pipeline.set_smooth_n(state.smooth_n)
                print(f"  Suavização: {state.smooth_n} frames")

            elif key in (ord('g'), ord('G')):
                state.hist_on = not state.hist_on

            elif key in (ord('l'), ord('L')):
                state.show_line_profile = not state.show_line_profile
                if not state.show_line_profile:
                    state.clear_line_pts()

            elif key == ord('k'):
                if calib.active:
                    calib.finalize()
                else:
                    calib.start()

            elif key == ord('K'):
                calib.reset()

            elif key in (ord('e'), ord('E')):
                readings = draw_probe_points(
                    out.copy(), enhanced, state, calib, split_offset)
                if readings:
                    calib.export_reading(readings)
                else:
                    print("  [CSV] Nenhum ponto de sonda ativo.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nCâmera encerrada.")


if __name__ == "__main__":
    main()
