"""
calibration.py — Calibragem por N pontos (regressão linear) com export CSV.

Melhorias sobre v2:
  • N pontos (mínimo 2) em vez de exatamente 2
  • Regressão linear por mínimos quadrados (numpy.polyfit) — mais robusta
  • Coeficiente de determinação R² exibido
  • Export de leituras para CSV com timestamp
  • pending_intensity como propriedade, sem mutação direta
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

CALIB_FILE = "calibration.json"
READINGS_CSV = "leituras_termica.csv"


@dataclass
class RefPoint:
    intensity: int
    temp_c: float


class TwoPointCalibration:
    """
    Calibragem linear por N pontos (N ≥ 2) via regressão de mínimos quadrados.

    Interface pública idêntica à v2 para retrocompatibilidade, com extras:
      • add_point(intensity)          — registra intensidade pendente (mouse)
      • consume_pending(img) → bool   — lê temperatura no terminal
      • is_ready() → bool
      • to_temp(intensity) → (val, unit)
      • export_csv(readings)          — salva leituras em CSV
    """

    MIN_POINTS = 2

    def __init__(self) -> None:
        self.coef_a: Optional[float] = None   # ganho  (°C / unidade)
        self.coef_b: Optional[float] = None   # offset (°C)
        self.r_squared: Optional[float] = None
        self.active: bool = False
        self.ref_pts: List[RefPoint] = []
        self._pending_intensity: Optional[int] = None
        self._load()

    # ── propriedade segura ───────────────────────────────────────────────────
    @property
    def pending_intensity(self) -> Optional[int]:
        return self._pending_intensity

    # ── persistência ────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not os.path.exists(CALIB_FILE):
            return
        try:
            with open(CALIB_FILE) as f:
                d = json.load(f)
            self.coef_a = d["coef_a"]
            self.coef_b = d["coef_b"]
            self.r_squared = d.get("r_squared")
            r2_str = f"  R²={self.r_squared:.4f}" if self.r_squared else ""
            print(f"  [CALIB] Carregada: T = {self.coef_a:.4f}·I + "
                  f"{self.coef_b:.2f} °C{r2_str}")
        except Exception as exc:
            print(f"  [CALIB] Erro ao carregar: {exc}")

    def _save(self) -> None:
        d = {
            "coef_a": self.coef_a,
            "coef_b": self.coef_b,
            "r_squared": self.r_squared,
            "ref_points": [{"intensity": p.intensity, "temp_c": p.temp_c}
                           for p in self.ref_pts],
        }
        with open(CALIB_FILE, "w") as f:
            json.dump(d, f, indent=2)
        print(f"  [CALIB] Salvo em {CALIB_FILE}")

    # ── conversão ───────────────────────────────────────────────────────────
    def is_ready(self) -> bool:
        return self.coef_a is not None

    def to_temp(self, intensity: int) -> Tuple[float, str]:
        """Retorna (valor, unidade). Sem calibragem → percentual 0-100 %."""
        if self.is_ready():
            return float(self.coef_a * intensity + self.coef_b), "°C"
        return intensity / 255.0 * 100.0, "%"

    # ── modo interativo ──────────────────────────────────────────────────────
    def start(self) -> None:
        self.active = True
        self.ref_pts = []
        self._pending_intensity = None
        print("\n  [CALIB] === MODO CALIBRAGEM ===")
        print("  Clique em 2+ pontos de referência na imagem (FRIO → QUENTE).")
        print("  Digite a temperatura real no terminal após cada clique.")
        print("  Pressione K novamente para finalizar com os pontos coletados,")
        print("  ou Shift+K para cancelar.\n")

    def cancel(self) -> None:
        self.active = False
        self.ref_pts = []
        self._pending_intensity = None
        print("  [CALIB] Cancelada.")

    def add_point(self, intensity: int) -> None:
        """Registra intensidade capturada pelo clique do mouse."""
        if self._pending_intensity is None:
            self._pending_intensity = intensity

    def consume_pending(self, img: np.ndarray) -> bool:
        """
        Lê temperatura no terminal para o ponto pendente.
        Retorna True se a calibragem foi concluída (≥ MIN_POINTS e K pressionado).
        Chamado no loop principal — não bloqueia o callback do mouse.
        """
        intensity = self._pending_intensity
        self._pending_intensity = None

        idx = len(self.ref_pts) + 1
        self._show_capture_overlay(img, intensity, idx)

        try:
            temp_c = float(input(
                f"  [CALIB] Ponto {idx}: intensidade={intensity} → temperatura [°C]: "))
        except (ValueError, EOFError):
            print("  [CALIB] Valor inválido, tente novamente.")
            return False

        self.ref_pts.append(RefPoint(intensity=intensity, temp_c=temp_c))
        print(f"  [CALIB] Ponto {idx} registrado: {intensity} → {temp_c:.1f}°C")
        print(f"  [CALIB] Continue clicando ou pressione K para finalizar "
              f"({len(self.ref_pts)} ponto(s) coletado(s)).")
        return False  # aguarda K para finalizar

    def finalize(self) -> bool:
        """Finaliza a calibragem com os pontos coletados até agora."""
        if len(self.ref_pts) < self.MIN_POINTS:
            print(f"  [CALIB] Mínimo de {self.MIN_POINTS} pontos necessário "
                  f"(coletados: {len(self.ref_pts)}).")
            return False
        return self._compute()

    def _compute(self) -> bool:
        intensities = np.array([p.intensity for p in self.ref_pts], dtype=float)
        temps = np.array([p.temp_c for p in self.ref_pts], dtype=float)

        if np.all(intensities == intensities[0]):
            print("  [CALIB] Todos os pontos têm a mesma intensidade — impossível calibrar.")
            self.cancel()
            return False

        # Regressão linear (grau 1) — mínimos quadrados
        coeffs = np.polyfit(intensities, temps, 1)
        self.coef_a = float(coeffs[0])
        self.coef_b = float(coeffs[1])

        # R²
        predicted = self.coef_a * intensities + self.coef_b
        ss_res = np.sum((temps - predicted) ** 2)
        ss_tot = np.sum((temps - np.mean(temps)) ** 2)
        self.r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0

        self.active = False
        self._save()
        print(f"\n  [CALIB] Concluída com {len(self.ref_pts)} ponto(s)!")
        print(f"          T = {self.coef_a:.4f} × I + {self.coef_b:.2f}  °C")
        print(f"          R² = {self.r_squared:.4f}\n")
        return True

    def reset(self) -> None:
        self.coef_a = self.coef_b = self.r_squared = None
        self.ref_pts = []
        self.active = False
        self._pending_intensity = None
        if os.path.exists(CALIB_FILE):
            os.remove(CALIB_FILE)
        print("  [CALIB] Removida.")

    # ── export CSV ───────────────────────────────────────────────────────────
    def export_reading(self, readings: list[dict]) -> None:
        """
        Adiciona leituras ao CSV.
        readings: lista de dicts com chaves intensity, temp_val, unit, px, py.
        """
        file_exists = os.path.exists(READINGS_CSV)
        with open(READINGS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "px", "py", "intensity",
                               "temp_val", "unit"])
            if not file_exists:
                writer.writeheader()
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            for r in readings:
                writer.writerow({"timestamp": ts, **r})
        print(f"  [CSV] {len(readings)} leitura(s) salvas em {READINGS_CSV}")

    # ── overlay de captura ───────────────────────────────────────────────────
    @staticmethod
    def _show_capture_overlay(img: np.ndarray, intensity: int, idx: int) -> None:
        overlay = img.copy()
        h, w = overlay.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        msgs = [
            (f"Ponto {idx} capturado  (intensidade={intensity})",
             h // 2 - 20, 0.65, (0, 220, 255)),
            ("Digite a temperatura no TERMINAL e pressione Enter...",
             h // 2 + 15, 0.45, (200, 200, 200)),
        ]
        for text, y, scale, color in msgs:
            (tw, _), _ = cv2.getTextSize(text, font, scale, 1)
            cv2.putText(img, text, ((w - tw) // 2, y),
                        font, scale, color, 1, cv2.LINE_AA)
        cv2.imshow("Camera Termica IR v3", img)
        cv2.waitKey(1)
