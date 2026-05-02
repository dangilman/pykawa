import numpy as np
from pykawa.units import k_GeV_per_v, hbarc, mchigram
from scipy.special import eval_legendre

def _prefactor(v):
    """4π/k² in cm²/g units."""
    k_GeV = v * k_GeV_per_v
    return 4 * np.pi * (hbarc / k_GeV)**2 / mchigram

def partial_wave_sum_angular(v_array, theta, phase_shifts_array):
    """
    Compute the differential cross section dσ/dΩ for arrays of velocities and phase shifts.
    :param v_array:           array of velocities in km/s, shape (n_v,)
    :param theta:             array of angles in radians, shape (n_theta,)
    :param phase_shifts_array: 2D array of phase shifts, shape (n_v, lconv+1)
                               each row is [δ_0, δ_1, ..., δ_lconv] for that velocity
    :returns:                 dσ/dΩ in cm²/g/sr, shape (n_v, n_theta)
    """
    v_array = np.atleast_1d(v_array)
    cos_theta = np.cos(theta)
    n_v = len(v_array)
    n_theta = len(theta)
    dsigma = np.zeros((n_v, n_theta))
    for i, (v, phase_shifts) in enumerate(zip(v_array, phase_shifts_array)):
        f_theta = np.zeros(n_theta, dtype=complex)
        for l, dl in enumerate(phase_shifts):
            if dl == 0 and l > 0:
                break  # stop at first zero-padded entry
            Pl = eval_legendre(l, cos_theta)
            f_theta += (2*l + 1) * np.exp(1j*dl) * np.sin(dl) * Pl
        k_GeV = v * k_GeV_per_v
        dsigma[i, :] = np.abs(f_theta)**2 * (hbarc / k_GeV)**2 / mchigram
    return dsigma

def partial_wave_sum_total(v_array, phase_shifts_array):
    """
    Total cross section.
    σ_tot = (4π/k²) Σ_l (2l+1) sin²(δ_l)
    :param v_array:            velocities in km/s, shape (n_v,)
    :param phase_shifts_array: phase shifts, shape (n_v, lmax+1), zero-padded
    :returns:                  σ_tot in cm²/g, shape (n_v,)
    """
    v_array = np.atleast_1d(v_array)
    result = np.zeros(len(v_array))
    for i, (v, ph) in enumerate(zip(v_array, phase_shifts_array)):
        lmax = np.nonzero(ph)[0][-1]
        s = sum((2*l + 1) * np.sin(ph[l])**2 for l in range(0, lmax))
        result[i] = _prefactor(v) * s
    return result

def partial_wave_sum_momentum(v_array, phase_shifts_array):
    """
    Momentum transfer cross section.
    σ_T = (4π/k²) Σ_l (l+1) sin²(δ_{l+1} - δ_l)
    :param v_array:            velocities in km/s, shape (n_v,)
    :param phase_shifts_array: phase shifts, shape (n_v, lmax+1), zero-padded
    :returns:                  σ_T in cm²/g, shape (n_v,)
    """
    v_array = np.atleast_1d(v_array)
    result = np.zeros(len(v_array))
    pad_zeros = 1
    zero_pad = [0.0] * pad_zeros
    for i, (v, ph) in enumerate(zip(v_array, phase_shifts_array)):
        lmax = np.nonzero(ph)[0][-1] + pad_zeros
        s = sum((l + 1) * np.sin(ph[l+1] - ph[l])**2 for l in range(0, lmax))
        result[i] = _prefactor(v) * s
    return result

def partial_wave_sum_viscosity(v_array, phase_shifts_array):
    """
    Viscosity transfer cross section.
    σ_V = (4π/k²) Σ_l (l+1)(l+2)/(2l+3) sin²(δ_{l+2} - δ_l)
    :param v_array:            velocities in km/s, shape (n_v,)
    :param phase_shifts_array: phase shifts, shape (n_v, lmax+1), zero-padded
    :returns:                  σ_V in cm²/g, shape (n_v,)
    """
    v_array = np.atleast_1d(v_array)
    result = np.zeros(len(v_array))
    pad_zeros = 2
    zero_pad = [0.0] * pad_zeros
    for i, (v, ph) in enumerate(zip(v_array, phase_shifts_array)):
        lmax = np.nonzero(ph)[0][-1] + pad_zeros
        ph_pad = np.append(ph, zero_pad)
        s = sum((l+1)*(l+2)/(2*l+3) * np.sin(ph_pad[l+2] - ph_pad[l])**2
                for l in range(0, lmax-1))
        result[i] = _prefactor(v) * s
    return result
