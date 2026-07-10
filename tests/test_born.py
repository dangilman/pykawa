"""Tests that validate numerical calculations against analytic solutions using Born approx.

In the dimensionless variables of the phase-shift solver (kappa = m_chi v / (2 m_phi c),
beta = 2 alpha m_phi c^2 / (m_chi v^2), w = 1/(2 kappa^2)), the raw partial-wave
sums S (such that sigma = prefactor(v) * S in pykawa's units) are:

    S_V^Born   = beta^2 kappa^2 [ (1+w) ln(1+2/w) - 2 ]
    S_T^Born   = beta^2 kappa^2 [ ln(1+2/w) - 2/(2+w) ] / 2
    S_tot^Born = beta^2 kappa^2 / ( w (2+w) )

derived from dsigma/dOmega = 4 beta^2 kappa^4 / (q^2 + 1)^2 with
q^2 = 2 kappa^2 (1 - cos theta).
"""
import numpy as np
import pytest
import numpy.testing as npt
from pykawa.utils import get_phases          # adjust import paths
from pykawa.cross_sections import CrossSection        # to match the package
from pykawa.partial_wave_sums import _prefactor, partial_wave_sum_angular
from pykawa.units import c_kms


class TestBorn(object):

    def setup_method(self):
        # Weakest-coupling corner of the repulsive grid: b = 10^(2.8-4.4) ~ 0.025.
        # Born relative error ~ |delta_0| < 2% everywhere on this model's grid;
        # the dominant residual is 4-decimal phase rounding at the low-v rows.
        self.BORN_MODEL = dict(log10_mass_ratio=2.8, log10alpha=-4.4,
                          potential='REPULSIVE_YUKAWA')

    @staticmethod
    def _kappa_beta(log10_mass_ratio, log10alpha, v):
        ratio = 10.0 ** log10_mass_ratio
        alpha = 10.0 ** log10alpha
        kappa = ratio * v / (2.0 * c_kms)
        beta = 2.0 * alpha * c_kms ** 2 / (ratio * v ** 2)
        return kappa, beta

    def _born_result(self, log10_mass_ratio, log10alpha, v, cross_section_type):
        """Closed-form solution of the cross section strength"""
        # NOTE: v must be float64. The npz grids are stored as float32, and the
        # VISCOSITY bracket (1+w)*log1p(2/w) - 2 is a catastrophic cancellation
        # at large w (~3e-6 extracted from numbers of size 2); float32 argument
        # error amplified by (1+w) ~ 450 renders it garbage (even negative) at
        # the lowest velocity rows.
        kappa, beta = self._kappa_beta(log10_mass_ratio, log10alpha, v)
        w = 1.0 / (2.0 * kappa ** 2)
        pref = _prefactor(v) * (beta * kappa) ** 2
        if cross_section_type == 'VISCOSITY':
            return pref * ((1.0 + w) * np.log1p(2.0 / w) - 2.0)
        if cross_section_type == 'MOMENTUM':
            return pref * (np.log1p(2.0 / w) - 2.0 / (2.0 + w)) / 2.0
        if cross_section_type == 'TOTAL':
            return pref / (w * (2.0 + w))
        raise ValueError(cross_section_type)

    @pytest.mark.parametrize("cs_type,tol_upper,tol_all",
                             [('VISCOSITY', 0.05, 0.12),
                              ('MOMENTUM', 0.05, 0.12),
                              ('TOTAL', 0.05, 0.12)])
    def test_solution(self, cs_type, tol_upper, tol_all):
        """Cross sections built from the stored phases must reproduce the
        closed-form Born results in the weak-coupling regime.

        Two-tier tolerance: the upper half of the velocity grid measures
        < 4% (best rows < 1%); the lowest rows degrade to ~8% purely from
        the 4-decimal rounding of phases of size ~1e-3, so a single tight
        tolerance would fail there for reasons unrelated to the code.
        """
        lmr = self.BORN_MODEL['log10_mass_ratio']
        la = self.BORN_MODEL['log10alpha']
        log10_v_grid, phases = get_phases(lmr, la, self.BORN_MODEL['potential'])
        cross = CrossSection.from_phase_shifts(log10_v_grid, phases, cs_type,
                                               la, lmr)
        v = 10.0 ** np.asarray(log10_v_grid, dtype=np.float64)
        numeric = cross(v, None)
        analytic = self._born_result(lmr, la, v, cs_type)
        rel = np.abs(numeric / analytic - 1.0)
        n_half = len(v) // 2
        assert np.all(np.isfinite(numeric)) and np.all(numeric > 0)
        npt.assert_array_less(rel[n_half:], tol_upper)
        npt.assert_array_less(rel, tol_all)

    def test_differential_cross_section_born(self):
        """The coherent partial-wave amplitude must reconstruct the Born
        angular shape dsigma/dOmega ~ (q^2 + 1)^-2 at all angles.

        This is the only test in the set sensitive to the sign conventions
        and complex structure of the amplitude: a wrong sign in exp(i*delta)
        or a dropped (2l+1) leaves the angular-moment tests untouched at
        weak coupling but shifts the interference pattern here.

        Restricted to mid-grid velocity rows (below, phases are smaller and
        rounding worsens; the top row adds the second-Born correction in the
        backward hemisphere, measured 18% at cos(theta) = -1). Within rows
        10-20 the measured envelope is 0.102, stochastic across rows and
        angles: the 4-decimal rounding of delta_0 (~1.5e-3 quantized to
        +/-5e-5) enters the coherent forward amplitude at the +/-3% level,
        giving +/-6% in |f|^2, with comparable backward excursions from
        Legendre-sum cancellations. Tolerance 0.12 = envelope + margin.
        """
        lmr = self.BORN_MODEL['log10_mass_ratio']
        la = self.BORN_MODEL['log10alpha']
        log10_v_grid, phases = get_phases(lmr, la, self.BORN_MODEL['potential'])
        theta = np.linspace(0.05, np.pi - 0.05, 60)
        rows = slice(10, 20)
        v = 10.0 ** np.asarray(log10_v_grid, dtype=np.float64)[rows]

        numeric = partial_wave_sum_angular(v, theta, phases[rows])

        kappa, beta = self._kappa_beta(lmr, la, v)
        q2 = 2.0 * kappa[:, None] ** 2 * (1.0 - np.cos(theta)[None, :])
        analytic = (_prefactor(v)[:, None] / (4.0 * np.pi)
                    * (2.0 * beta[:, None] * kappa[:, None] ** 3) ** 2
                    / (q2 + 1.0) ** 2)

        rel = np.abs(numeric / analytic - 1.0)
        # Two-tier: quantization jitter produces occasional ~10% spikes at the
        # theta endpoints (measured max 0.102, rows 11/19) while the bulk sits
        # at the percent level (measured median ~0.02). A genuine amplitude bug
        # shifts the whole distribution, so the tight median bound is the
        # discriminating assertion; the ceiling catches gross regressions.
        assert np.median(rel) < 0.04, f"median angular rel dev {np.median(rel):.3f}"
        npt.assert_array_less(rel, 0.12)

    def test_phase_shifts_born_tail(self):
        """Stored phase shifts must match the analytic per-partial-wave Born
        formula delta_l = -beta*kappa*Q_l(1 + 1/(2 kappa^2)) in the window
        1e-3 < |delta| < 2e-2 (below: rounding-dominated; above: second-Born
        correction exceeds the 5% tolerance)."""
        from scipy.integrate import quad

        def legendre_q(l, x):
            f = lambda t: (x + np.sqrt(x * x - 1.0) * np.cosh(t)) ** (-(l + 1))
            return quad(f, 0.0, 60.0, limit=200)[0]

        lmr = self.BORN_MODEL['log10_mass_ratio']
        la = self.BORN_MODEL['log10alpha']
        log10_v_grid, phases = get_phases(lmr, la, self.BORN_MODEL['potential'])
        checked = 0
        for k in range(len(log10_v_grid)):
            v = 10.0 ** log10_v_grid[k]
            kappa, beta = self._kappa_beta(lmr, la, v)
            row = phases[k]
            for l in np.nonzero(row)[0]:
                if not (1e-3 < abs(row[l]) < 2e-2):
                    continue
                born = -beta * kappa * legendre_q(int(l),
                                                  1.0 + 1.0 / (2 * kappa ** 2))
                npt.assert_allclose(row[l], born, rtol=0.05,
                                    err_msg=f"row {k}, l={l}")
                checked += 1
        assert checked > 10, "too few phases in the testable window"


if __name__ == "__main__":
    pytest.main()
