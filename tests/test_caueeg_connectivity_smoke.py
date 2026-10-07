from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from step18_caueeg_connectivity import aec_signed, aecc_signed, bandpass_sos, wpli_debiased  # noqa: E402

FS = 200.0


def _band_noise(rng, n, band=(8.0, 12.0)):
    return bandpass_sos(rng.standard_normal((1, n)), band, FS)[0]


def test_zero_lag_mixing_inflates_aec_but_not_aecc_or_wpli():
    rng = np.random.default_rng(0)
    n = 20000
    source = _band_noise(rng, n) * (1 + 0.8 * np.sin(np.arange(n) / 400.0))
    noise = np.stack([_band_noise(rng, n) for _ in range(3)])
    # Volume conduction: the same source enters every channel with zero lag.
    data = np.stack([source + 0.3 * noise[0], 0.8 * source + 0.3 * noise[1], noise[2]])
    analytic = hilbert(data, axis=1)

    aec = aec_signed(analytic)
    aecc = aecc_signed(analytic)
    wpli = wpli_debiased(analytic)
    assert aec[0, 1] > 0.7
    assert abs(aecc[0, 1]) < 0.15
    assert abs(wpli[0, 1]) < 0.05
    for matrix in (aec, aecc, wpli):
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 0.0)


def test_lagged_coupling_is_detected_by_aecc_and_wpli():
    rng = np.random.default_rng(1)
    n = 20000
    analytic_source = hilbert(_band_noise(rng, n))
    envelope = 1 + 0.9 * np.sin(np.arange(n) / 300.0)
    a = np.real(analytic_source) * envelope
    b = np.real(analytic_source * np.exp(1j * np.pi / 2)) * envelope  # 90 degree phase lag
    data = np.stack([a + 0.2 * _band_noise(rng, n), b + 0.2 * _band_noise(rng, n), _band_noise(rng, n)])
    analytic = hilbert(data, axis=1)

    assert aecc_signed(analytic)[0, 1] > 0.3
    assert wpli_debiased(analytic)[0, 1] > 0.8
    assert abs(wpli_debiased(analytic)[0, 2]) < 0.05
