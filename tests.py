"""
tests.py — Testes unitários para camera_termica_v3.

Cobertura:
  • TwoPointCalibration._compute / to_temp / reset
  • IRPipeline.detect_hotspots / line_profile / compute_stats
  • AppState helpers (probe_points, line_profile_pts)

Execute com:
    python tests.py
"""

import sys
import os
import unittest
import numpy as np

# Garante que o diretório do projeto está no path
sys.path.insert(0, os.path.dirname(__file__))

from calibration import TwoPointCalibration, RefPoint
from pipeline import IRPipeline, ThermalStats
from app_state import AppState


# ── Calibragem ────────────────────────────────────────────────────────────────
class TestCalibration(unittest.TestCase):

    def _calib_with_points(self, pts):
        """Cria uma instância de calibragem já com ref_pts definidos."""
        c = TwoPointCalibration.__new__(TwoPointCalibration)
        c.coef_a = None
        c.coef_b = None
        c.r_squared = None
        c.active = True
        c._pending_intensity = None
        c.ref_pts = [RefPoint(intensity=i, temp_c=t) for i, t in pts]
        return c

    def test_two_point_perfect(self):
        """Calibragem exata com 2 pontos: 0→0°C e 255→100°C."""
        c = self._calib_with_points([(0, 0.0), (255, 100.0)])
        c._save = lambda: None          # não grava arquivo em teste
        result = c._compute()
        self.assertTrue(result)
        self.assertAlmostEqual(c.to_temp(0)[0],   0.0, places=4)
        self.assertAlmostEqual(c.to_temp(255)[0], 100.0, places=4)
        self.assertAlmostEqual(c.r_squared,        1.0, places=4)

    def test_three_point_regression(self):
        """Regressão com 3 pontos alinhados — R² deve ser 1."""
        # T = 0.5 * I − 10  →  pontos: (20,0), (100,40), (200,90)
        c = self._calib_with_points([(20, 0.0), (100, 40.0), (200, 90.0)])
        c._save = lambda: None
        c._compute()
        self.assertAlmostEqual(c.r_squared, 1.0, places=3)
        self.assertAlmostEqual(c.to_temp(20)[0],  0.0, places=1)
        self.assertAlmostEqual(c.to_temp(200)[0], 90.0, places=1)

    def test_same_intensity_aborts(self):
        """Dois pontos com intensidade idêntica devem cancelar a calibragem."""
        c = self._calib_with_points([(128, 20.0), (128, 80.0)])
        c._save = lambda: None
        result = c._compute()
        self.assertFalse(result)
        self.assertIsNone(c.coef_a)

    def test_no_calibration_returns_percent(self):
        """Sem calibragem, to_temp deve retornar percentual 0-100 %."""
        c = TwoPointCalibration.__new__(TwoPointCalibration)
        c.coef_a = None
        c.coef_b = None
        val, unit = c.to_temp(128)
        self.assertEqual(unit, "%")
        self.assertAlmostEqual(val, 128 / 255 * 100, places=2)

    def test_reset_clears_coefficients(self):
        """reset() deve zerar coeficientes sem lançar exceção."""
        c = self._calib_with_points([(0, 0.0), (255, 100.0)])
        c._save = lambda: None
        c._compute()
        c.reset()
        self.assertIsNone(c.coef_a)
        self.assertIsNone(c.coef_b)
        self.assertFalse(c.is_ready())

    def test_unit_after_calibration(self):
        """to_temp deve retornar '°C' após calibragem bem-sucedida."""
        c = self._calib_with_points([(0, 20.0), (255, 80.0)])
        c._save = lambda: None
        c._compute()
        _, unit = c.to_temp(128)
        self.assertEqual(unit, "°C")

    def test_min_points_not_reached(self):
        """finalize() com 1 ponto deve retornar False."""
        c = self._calib_with_points([(100, 25.0)])
        c._save = lambda: None
        c.active = True
        result = c.finalize()
        self.assertFalse(result)


# ── Pipeline ──────────────────────────────────────────────────────────────────
class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = IRPipeline(alpha=3.0)

    # ── hotspots ──────────────────────────────────────────────────────────────
    def test_hotspot_detected_above_threshold(self):
        """Um quadrado brilhante deve ser detectado como hotspot."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[40:60, 40:60] = 240          # região quente
        spots = IRPipeline.detect_hotspots(gray, threshold=220, min_area=50)
        self.assertGreater(len(spots), 0)

    def test_no_hotspot_below_threshold(self):
        """Imagem toda abaixo do threshold não deve retornar nenhum hotspot."""
        gray = np.full((100, 100), 100, dtype=np.uint8)
        spots = IRPipeline.detect_hotspots(gray, threshold=220, min_area=50)
        self.assertEqual(len(spots), 0)

    def test_hotspot_centroid_location(self):
        """Centroide do hotspot deve estar próximo ao centro do quadrado."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[40:60, 40:60] = 240
        spots = IRPipeline.detect_hotspots(gray, threshold=220, min_area=50)
        cx, cy = spots[0][0], spots[0][1]
        self.assertAlmostEqual(cx, 49, delta=3)
        self.assertAlmostEqual(cy, 49, delta=3)

    # ── line profile ──────────────────────────────────────────────────────────
    def test_line_profile_length(self):
        """line_profile deve retornar exatamente n_samples pontos."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        profile = IRPipeline.line_profile(gray, (0, 0), (99, 99), n_samples=50)
        self.assertEqual(len(profile), 50)

    def test_line_profile_values_in_range(self):
        """Valores do perfil devem estar no intervalo [0, 255]."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        profile = IRPipeline.line_profile(gray, (10, 10), (90, 90))
        self.assertTrue(np.all(profile >= 0))
        self.assertTrue(np.all(profile <= 255))

    def test_line_profile_horizontal(self):
        """Perfil horizontal em linha constante deve ser uniforme."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[50, :] = 128
        profile = IRPipeline.line_profile(gray, (0, 50), (99, 50), n_samples=10)
        self.assertTrue(np.all(profile == 128))

    # ── estatísticas ──────────────────────────────────────────────────────────
    def test_stats_min_max(self):
        gray = np.zeros((50, 50), dtype=np.uint8)
        gray[10, 10] = 200
        gray[40, 40] = 5
        stats = IRPipeline.compute_stats(gray)
        self.assertEqual(stats.max_val, 200)
        self.assertEqual(stats.min_val, 0)

    def test_stats_hot_location(self):
        gray = np.zeros((50, 50), dtype=np.uint8)
        gray[20, 30] = 255
        stats = IRPipeline.compute_stats(gray)
        self.assertEqual(stats.hot_x, 30)
        self.assertEqual(stats.hot_y, 20)

    def test_stats_mean(self):
        gray = np.full((10, 10), 100, dtype=np.uint8)
        stats = IRPipeline.compute_stats(gray)
        self.assertAlmostEqual(stats.mean_val, 100.0, places=1)

    # ── CLAHE — re-instanciação ────────────────────────────────────────────────
    def test_clahe_recreated_on_alpha_change(self):
        """set_alpha deve atualizar o atributo alpha sem lançar exceção."""
        self.pipeline.set_alpha(5.0)
        self.assertAlmostEqual(self.pipeline.alpha, 5.0)
        self.pipeline.set_alpha(0.1)        # abaixo do mínimo → clampado
        self.assertAlmostEqual(self.pipeline.alpha, 0.5)


# ── AppState ──────────────────────────────────────────────────────────────────
class TestAppState(unittest.TestCase):

    def test_add_probe_max(self):
        """Não deve adicionar mais de MAX_PROBE_POINTS pontos."""
        state = AppState()
        for i in range(10):
            state.add_probe(i * 20, i * 20)
        from app_state import MAX_PROBE_POINTS
        self.assertLessEqual(len(state.probe_points), MAX_PROBE_POINTS)

    def test_remove_probe_on_nearby_click(self):
        """Clicar perto de um ponto existente deve removê-lo."""
        state = AppState()
        state.add_probe(50, 50)
        self.assertEqual(len(state.probe_points), 1)
        state.add_probe(52, 51)   # dentro do raio de 15px
        self.assertEqual(len(state.probe_points), 0)

    def test_clear_probes(self):
        state = AppState()
        state.add_probe(10, 10)
        state.add_probe(20, 20)
        state.clear_probes()
        self.assertEqual(len(state.probe_points), 0)

    def test_line_pts_resets_on_third_click(self):
        """Terceiro clique de linha deve reiniciar a seleção."""
        state = AppState()
        state.add_line_pt(0, 0)
        state.add_line_pt(100, 100)
        self.assertEqual(len(state.line_profile_pts), 2)
        state.add_line_pt(50, 50)   # terceiro: zera e adiciona
        self.assertEqual(len(state.line_profile_pts), 1)

    def test_fps_tick(self):
        """tick() não deve lançar exceção e fps deve ser ≥ 0."""
        state = AppState()
        for _ in range(20):
            state.tick()
        self.assertGreaterEqual(state.fps, 0.0)


if __name__ == "__main__":
    print("=== Testes camera_termica_v3 ===\n")
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(__import__("__main__"))
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
