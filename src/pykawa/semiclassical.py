"""
Semi-classical viscosity cross section for a Yukawa potential, distinguishable particles:
Colquhoun, Heeba, Kahlhoefer, Sagunski & Tulin (2021), arXiv:2011.04679, Eqs. (83)-(89).
sigma_V m_phi^2 / pi = f(beta, kappa), valid for kappa = m_chi v / (2 m_phi) >~ 1.

sigmaV_semiclassical returns sigma_V/m_chi in pykawa's units (cm^2/g for m_chi = 1 GeV), the same
normalization as the tabulated cross sections, so it can be combined directly with pykawa interpolators.
"""
import numpy as np
from scipy.special import k0, k1, kn
from pykawa.partial_wave_sums import _prefactor
from pykawa.units import c_kms

_LAMBDA_V = (9 - np.cos(4) - 4 * np.sin(4)) / 16


def _eta(x):
    return x ** 2 * (-k1(x) ** 2 + k0(x) * kn(2, x))


def _zeta(n, kappa, beta):
    m = np.maximum(n, beta * kappa)
    return (m ** 2 - n ** 2) / (2 * kappa ** 2 * beta ** 2) + _eta(m / kappa)


def fV(beta, kappa, attractive=True, n=1):
    """sigma_V m_phi^2 / pi (Eqs. 86-87); n = 1 for distinguishable particles"""
    b, kappa = np.broadcast_arrays(np.asarray(beta, float), np.asarray(kappa, float))
    with np.errstate(all='ignore'):
        weak = 4 * b ** 2 * _zeta(n, kappa, 2 * b)
        if attractive:
            return np.where(b <= 0.1, weak,
                   np.where(b <= 0.5, weak * np.exp(0.67 * (b - 0.1)),
                   np.where(b < 25, 2.5 * np.log(b + 1.05),
                            0.5 * (1 + np.log(b) - 1 / (2 * np.log(b))) ** 2)))
        L = np.log(2 * b)
        return np.where(b <= 0.1, weak,
               np.where(b <= 0.5, weak * np.exp(-0.37 * (b - 0.1)),
               np.where(b < 25, 2.8 * np.log(b + 0.80),
                        L * (_LAMBDA_V * L - (2 * _LAMBDA_V - 1) * np.log(L)))))


def sigmaV_semiclassical(v, log10alpha, log10_mass_ratio, attractive=True):
    """
    sigma_V/m_chi [cm^2/g] at m_chi = 1 GeV (pykawa's table normalization), for relative velocity v [km/s].
    Uses sigma_V = (4 pi / k^2) S with S = f(beta, kappa) kappa^2 / 4.
    """
    v = np.asarray(v, float)
    r = 10.0 ** np.asarray(log10_mass_ratio, float)
    kappa = r * v / (2 * c_kms)
    beta = 2 * 10.0 ** np.asarray(log10alpha, float) * c_kms ** 2 / (r * v ** 2)
    return _prefactor(v) * fV(beta, kappa, attractive) * kappa ** 2 / 4


def sigmaV_born(v, log10alpha, log10_mass_ratio):
    """First-order Born sigma_V/m_chi [cm^2/g] at m_chi = 1 GeV (same for attractive and repulsive)"""
    v = np.asarray(v, float)
    r = 10.0 ** np.asarray(log10_mass_ratio, float)
    kappa = r * v / (2 * c_kms)
    beta = 2 * 10.0 ** np.asarray(log10alpha, float) * c_kms ** 2 / (r * v ** 2)
    w = 1.0 / (2 * kappa ** 2)
    return _prefactor(v) * (beta * kappa) ** 2 * ((1 + w) * np.log1p(2 / w) - 2)


def sigmaV_high_velocity(v, log10alpha, log10_mass_ratio, attractive=True):
    """
    sigma_V/m_chi [cm^2/g] at m_chi = 1 GeV for semi-classical velocities (kappa >~ 1):
    Born where it is accurate (beta < 0.1 and kappa*beta = alpha c / v < 0.5), otherwise the
    semi-classical formulas of Colquhoun et al. (2021). Tested against the computed attractive table at
    1500 km/s: Born to ~4% (95% within 8%) in its region, semi-classical to ~10-15% elsewhere.
    """
    v = np.asarray(v, float)
    r = 10.0 ** np.asarray(log10_mass_ratio, float)
    kappa = r * v / (2 * c_kms)
    beta = 2 * 10.0 ** np.asarray(log10alpha, float) * c_kms ** 2 / (r * v ** 2)
    use_born = (beta < 0.1) & (kappa * beta < 0.5)
    return np.where(use_born, sigmaV_born(v, log10alpha, log10_mass_ratio),
                    sigmaV_semiclassical(v, log10alpha, log10_mass_ratio, attractive))
