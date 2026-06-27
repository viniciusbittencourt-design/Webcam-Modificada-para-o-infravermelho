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

import argparse
import sys
import time

import cv2

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

WINDOW_MAIN = "Camera Termica IR v3"
WINDOW_HIST = "Histograma IR"
WINDOW_PROFILE = "Line Profile"


# ── Argumentos ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Câmera térmica IR com análise avançada (v3)")
    p.add_argument("--camera",          type=int, default=0)
    p.add_argument("--smooth",          type=int, default=5,
                   help="Frames para suavização temporal (1=desligado)")
    p.add_argument("--hotspot-thresh",  type=int, default=220,
                   help="Limiar de intensidade para hotspot (0-255)")
    return p.parse_args()


# ── Mouse callback (sem globals) ─────────────────────────────────────────────
def make_mouse_callback(state: AppState, calib: TwoPointCalibration):
    """
    Retorna um callback de mouse que fecha sobre state e calib.
    Nenhuma variável global é necessária.
    """
    shift_held = [False]   # workaround: flags de teclado não chegam pelo mouse

    def callback(event, x, y, flags, param):
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)

        # ── durante calibragem ───────────────────────────────────────────────
        if event == cv2.EVENT_LBUTTONDOWN and calib.active:
            if calib.pending_intensity is None and state.last_enhanced is not None:
                enhanced = state.last_enhanced
                gx = x - (state.frame_w if state.split_view else 0)
                gy = y
                gx = max(0, min(gx, enhanced.shape[1] - 1))
                gy = max(0, min(gy, enhanced.shape[0] - 1))
                calib.add_point(int(enhanced[gy, gx]))
            return

        # ── line profile (Shift + clique-esq) ───────────────────────────────
        if event == cv2.EVENT_LBUTTONDOWN and shift:
            state.add_line_pt(x, y)
            return

        # ── sondas (clique-esq normal) ───────────────────────────────────────
        if event == cv2.EVENT_LBUTTONDOWN:
            state.add_probe(x, y)
            return

        # ── limpar sondas (clique-dir) ───────────────────────────────────────
        if event == cv2.EVENT_RBUTTONDOWN:
            state.clear_probes()

    return callback


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"\n[ERRO] Não foi possível abrir a câmera {args.camera}.")
        print("  → Tente: python camera_termica_v3.py --camera 1")
        sys.exit(1)

    # Solicita a resolução máxima que o driver aceitar.
    # Valores absurdamente altos fazem o driver negociar para o maior modo
    # suportado pela câmera, sem forçar um valor específico.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  10_000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10_000)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  [CAM] Resolução negociada: {W}x{H}")

    # ── instâncias ───────────────────────────────────────────────────────────
    state = AppState(
        camera_idx=args.camera,
        frame_w=W,
        frame_h=H,
        smooth_n=args.smooth,
        hotspot_thresh=args.hotspot_thresh,
    )
    pipeline  = IRPipeline(alpha=state.alpha)
    pipeline.set_smooth_n(state.smooth_n)
    calib     = TwoPointCalibration()
    recorder  = VideoRecorder(W, H)

    # ── janela e callbacks ───────────────────────────────────────────────────
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, W, H)
    cv2.setMouseCallback(WINDOW_MAIN, make_mouse_callback(state, calib))

    print("\n=== CÂMERA TÉRMICA IR v3 ===")
    print("  Q / ESC  — Sair")
    print("  S        — Salvar foto PNG")
    print("  R        — Iniciar/parar gravação")
    print("  C        — Trocar colormap")
    print("  +/-      — Contraste CLAHE")
    print("  V        — Split-view (original | térmico)")
    print("  H        — Hotspot detector")
    print("  T        — Mais/menos suavização temporal (ciclo 1-15)")
    print("  G        — Histograma ao vivo")
    print("  L        — Mostrar/ocultar line profile")
    print("  K        — Iniciar calibragem / finalizar coleta")
    print("  Shift+K  — Resetar calibragem")
    print("  E        — Exportar leituras atuais para CSV")
    print("  Clique-esq       — Adicionar ponto de sonda")
    print("  Shift+Clique     — Definir ponto do line profile (2 cliques = linha)")
    print("  Clique-dir       — Limpar pontos de sonda\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        state.tick()

        out, enhanced, split_offset, stats = pipeline.process(frame, state)
        state.last_enhanced = enhanced

        # ── overlays ─────────────────────────────────────────────────────────
        draw_stats_panel(out, stats, calib, split_offset)
        readings = draw_probe_points(out, enhanced, state, calib, split_offset)
        draw_magnifier(out, stats.hot_x + split_offset, stats.hot_y)
        draw_calibration_overlay(out, calib)
        draw_hud(out, state, calib, recorder)

        # ── gravação ──────────────────────────────────────────────────────────
        if recorder.active:
            recorder.write(out)
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(out, (0, 0),
                              (out.shape[1] - 1, out.shape[0] - 1),
                              (0, 0, 200), 4)

        cv2.imshow(WINDOW_MAIN, out)

        # ── calibragem: consumir ponto pendente ───────────────────────────────
        if calib.active and calib.pending_intensity is not None:
            calib.consume_pending(out)

        # ── histograma ────────────────────────────────────────────────────────
        if state.hist_on:
            hist_img = draw_histogram(enhanced, state.colormap_idx)
            cv2.imshow(WINDOW_HIST, hist_img)
        else:
            try:
                cv2.destroyWindow(WINDOW_HIST)
            except Exception:
                pass

        # ── line profile ──────────────────────────────────────────────────────
        if state.show_line_profile and len(state.line_profile_pts) == 2:
            profile = pipeline.line_profile(
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

        # ── teclado ───────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            if recorder.active:
                recorder.stop()
            break

        elif key in (ord('s'), ord('S')):
            fname = f"termica_{int(time.time())}.png"
            cv2.imwrite(fname, out)
            print(f"  [FOTO] {fname}")

        elif key in (ord('r'), ord('R')):
            if recorder.active:
                recorder.stop()
            else:
                recorder.start()

        elif key in (ord('c'), ord('C')):
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
            print(f"  Suavização temporal: {state.smooth_n} frames")

        elif key in (ord('g'), ord('G')):
            state.hist_on = not state.hist_on

        elif key in (ord('l'), ord('L')):
            state.show_line_profile = not state.show_line_profile
            if not state.show_line_profile:
                state.clear_line_pts()

        elif key == ord('k'):   # minúsculo: iniciar ou finalizar calibragem
            if calib.active:
                calib.finalize()   # ← novo método: finaliza com N pontos
            else:
                calib.start()

        elif key == ord('K'):   # maiúsculo (Shift+K): resetar
            calib.reset()

        elif key in (ord('e'), ord('E')):
            if readings:
                calib.export_reading(readings)
            else:
                print("  [CSV] Nenhum ponto de sonda ativo.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nCâmera encerrada.")


if __name__ == "__main__":
    main()
