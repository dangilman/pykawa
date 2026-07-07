import numpy as np
from pykawa.units import k_GeV_per_v, hbarc, mchigram
from scipy.special import eval_legendre

def partial_wave_sum_angular(v_array, theta, phase_shifts_array):
    """Differential cross section dsigma/dOmega.

    Phases enter individually (no differences), so zero-padded entries
    contribute nothing and no truncation logic is needed.

    :param v_array:            velocities in km/s, shape (n_v,)
    :param theta:              angles in radians, shape (n_theta,)
    :param phase_shifts_array: phase shifts, shape (n_v, lmax+1), zero-padded
    :returns:                  dsigma/dOmega in cm^2/g/sr, shape (n_v, n_theta)
    """
    v_array = np.atleast_1d(v_array)
    cos_theta = np.cos(theta)
    dsigma = np.zeros((len(v_array), len(theta)))
    for i, (v, ph) in enumerate(zip(v_array, phase_shifts_array)):
        f_theta = np.zeros(len(theta), dtype=complex)
        for l in np.nonzero(ph)[0]:
            dl = ph[l]
            f_theta += (2 * l + 1) * np.exp(1j * dl) * np.sin(dl) * eval_legendre(l, cos_theta)
        k_GeV = v * k_GeV_per_v
        dsigma[i, :] = np.abs(f_theta) ** 2 * (hbarc / k_GeV) ** 2 / mchigram
    return dsigma

def partial_wave_sum_total(v_array, phase_shifts_array):
    """Total cross section, sigma_tot = (4pi/k^2) sum_l (2l+1) sin^2(delta_l).

    Phases enter individually (no differences), so zero-padded entries
    contribute sin^2(0) = 0 and no truncation logic is needed.

    :param v_array:            velocities in km/s, shape (n_v,)
    :param phase_shifts_array: phase shifts, shape (n_v, lmax+1), zero-padded
    :returns:                  sigma_tot in cm^2/g, shape (n_v,)
    """
    v_array = np.atleast_1d(v_array)
    result = np.zeros(len(v_array))
    for i, (v, ph) in enumerate(zip(v_array, phase_shifts_array)):
        l = np.arange(len(ph))
        result[i] = _prefactor(v) * np.sum((2 * l + 1) * np.sin(ph) ** 2)
    return result

# keep all phase shifts up to l = 14, as these were computed for all models
_L_FLOOR = 14

def partial_wave_sum_viscosity(v_array, phase_shifts_array):
    v_array = np.atleast_1d(v_array)
    result = np.zeros(len(v_array))
    for i, (v, ph) in enumerate(zip(v_array, phase_shifts_array)):
        n = len(ph) - 2
        l = np.arange(n)
        trust_lo = (ph[:-2] != 0) | (l <= _L_FLOOR)
        trust_hi = (ph[2:] != 0) | (l + 2 <= _L_FLOOR)
        mask = trust_lo & trust_hi & ((ph[:-2] != 0) | (ph[2:] != 0))
        s = np.sum(((l + 1) * (l + 2) / (2 * l + 3))[mask]
                   * np.sin(ph[2:][mask] - ph[:-2][mask]) ** 2)
        result[i] = _prefactor(v) * s
    return result

def partial_wave_sum_momentum(v_array, phase_shifts_array):
    v_array = np.atleast_1d(v_array)
    result = np.zeros(len(v_array))
    for i, (v, ph) in enumerate(zip(v_array, phase_shifts_array)):
        n = len(ph) - 1
        l = np.arange(n)
        trust_lo = (ph[:-1] != 0) | (l <= _L_FLOOR)
        trust_hi = (ph[1:] != 0) | (l + 1 <= _L_FLOOR)
        mask = trust_lo & trust_hi & ((ph[:-1] != 0) | (ph[1:] != 0))
        s = np.sum((l + 1)[mask] * np.sin(ph[1:][mask] - ph[:-1][mask]) ** 2)
        result[i] = _prefactor(v) * s
    return result

def _prefactor(v):
    """4pi/k^2 in cm^2/g units."""
    k_GeV = v * k_GeV_per_v
    return 4 * np.pi * (hbarc / k_GeV) ** 2 / mchigram

def _l_trust(ph, lconv=None, eps=0.05):
    """Highest l index whose phase (including physical zeros) may enter pair sums.
    """
    nz = np.nonzero(ph)[0]
    if len(nz) == 0:
        return 0
    L = int(nz[-1])
    if lconv is not None:
        return int(lconv)
    return L + 2 if abs(ph[L]) < eps else L
