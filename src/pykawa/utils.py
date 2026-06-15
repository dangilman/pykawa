from scipy.integrate import cumulative_trapezoid
import numpy as np
from importlib.resources import files


def extrapolate_cross_section(v, sigma_v0, slope_at_v0, slope_of_slope, slope_end=-4.0):
    """
    Extrapolate σ(v) beyond the tabulated velocity range by integrating a power law
    whose logarithmic slope transitions linearly from slope_at_v0 toward slope_end.

    The logarithmic slope d(log σ)/d(log v) is modelled as:
        exponent(v) = max(slope_at_v0 + slope_of_slope * log(v/v0), slope_end)

    and σ(v) is recovered by exponentiating the cumulative integral:
        log σ(v) = log σ(v0) + ∫_{v0}^{v} exponent(v') d(log v')

    This allows a smooth transition from the slope at the last tabulated point to
    the asymptotic high-velocity power law σ ∝ v^slope_end.

    :param v:              velocity array for the extrapolation region, shape (n,)
                           with v[0] equal to the last tabulated velocity
    :param sigma_v0:       cross section at v[0] in cm²/g (the anchor point)
    :param slope_at_v0:    initial logarithmic slope d(log σ)/d(log v) at v[0],
                           typically obtained from get_slope_at_vmatch()
    :param slope_of_slope: rate at which the slope changes with log(v/v0);
                           a negative value causes the slope to steepen with velocity
    :param slope_end:      asymptotic logarithmic slope at high velocity (default -4),
                           corresponding to σ ∝ v^{-4} in the geometric regime
    :returns:              extrapolated cross section array in cm²/g, shape (n,)
    """
    v_0 = v[0]
    log_x = np.log(v / v_0)

    expon = slope_at_v0 + slope_of_slope * log_x
    expon = np.maximum(expon, slope_end)

    log_sigma = np.zeros(len(v))
    log_sigma[1:] = cumulative_trapezoid(expon, log_x)
    return sigma_v0 * np.exp(log_sigma)


def get_slope_at_vmatch(v, sigma_v):
    """
    Estimate the logarithmic slope d(log σ)/d(log v) at the last tabulated velocity,
    using a numerical gradient over the full velocity array. This slope is used as the
    initial condition for the high-velocity extrapolation in extrapolate().

    :param v:       velocity array in km/s, shape (n_v,)
    :param sigma_v: cross section array in cm²/g, shape (n_v,)
    :returns:       scalar slope d(log σ)/d(log v) evaluated at v[-1]
    """
    dlogsigma_dlogv = np.gradient(np.log(sigma_v), np.log(v))
    return dlogsigma_dlogv[-1]

_cache = {}
def get_phases(log10_mass_ratio, log10_alpha, potential='REPULSIVE_YUKAWA'):
    """
    Return the velocity grid and phase shifts for a given parameter combination.

    The precomputed grids cover the following ranges:

        REPULSIVE_YUKAWA:
            log10_mass_ratio :  2.8 –  5.2, step 0.2  (13 values)
            log10_alpha      : -2.0 – -4.4, step -0.3  (9 values)

        ATTRACTIVE_YUKAWA:
            log10_mass_ratio :  3.5 –  4.7, step 0.03  (41 values)
            log10_alpha      : -2.6 – -4.6, step -0.05 (41 values)

    Parameters
    ----------
    log10_mass_ratio : float
        log10(m_chi / m_phi). Must match a value in the precomputed grid.
    log10_alpha : float
        log10(coupling), expected to be negative. Must match a value in
        the precomputed grid.
    potential : str, optional
        Either 'ATTRACTIVE_YUKAWA' or 'REPULSIVE_YUKAWA'.
        Default is 'REPULSIVE_YUKAWA'.

    Returns
    -------
    v : ndarray, shape (N_v,)
        Velocity grid. The velocity grid differs for each parameter combination.
    phases : ndarray, shape (N_v, N_ell)
        Phase shifts delta_ell(v), with ell running from 0 to N_ell - 1.
        Computed to l_max = 300. Columns beyond the converged ell are zero.

    Raises
    ------
    ValueError
        If log10_mass_ratio or log10_alpha is not in the precomputed grid.
    """
    scale = 10 if potential == 'REPULSIVE_YUKAWA' else 100
    mphi_key  = int(round(log10_mass_ratio * scale))
    alpha_key = int(round(-log10_alpha * scale))  # stored as positive integers

    if potential not in _cache:
        path = files("pykawa.data").joinpath(f"{potential}_phases.npz")
        _cache[potential] = np.load(path)

    data = _cache[potential]
    mphi_vals  = data["mphi"]
    alpha_vals = data["alpha"]

    i = np.searchsorted(mphi_vals, mphi_key)
    if i >= len(mphi_vals) or mphi_vals[i] != mphi_key:
        raise ValueError(
            f"log10_mass_ratio={log10_mass_ratio} not in precomputed grid. "
            f"Available: {(mphi_vals / scale).tolist()}"
        )

    j = np.searchsorted(alpha_vals, alpha_key)
    if j >= len(alpha_vals) or alpha_vals[j] != alpha_key:
        raise ValueError(
            f"log10_alpha={log10_alpha} not in precomputed grid. "
            f"Available: {(-alpha_vals / scale).tolist()}"
        )

    return data["v"][i, j], data["phases"][i, j]
