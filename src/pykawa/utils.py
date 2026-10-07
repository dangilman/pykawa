from scipy.integrate import cumulative_trapezoid
import numpy as np
from importlib.resources import files


def extrapolate_cross_section(v, sigma_v0, slope_at_v0, slope_of_slope, slope_end=-4.0):
    """
    Extrapolate sigma(v) beyond the tabulated range by continuing the log-slope
    with the curvature it exhibits at v[0], until it reaches slope_end:

        exponent(v) = max(slope_at_v0 + slope_of_slope * log(v/v0), slope_end)

    :param v:              extrapolation velocities, shape (n,); v[0] = last tabulated v
    :param sigma_v0:       cross section at v[0] in cm^2/g
    :param slope_at_v0:    log-slope at v[0] (from get_slope_at_vmatch)
    :param slope_of_slope: d^2(log sigma)/d(log v)^2 at v[0]
                           (from get_slope_curvature_at_vmatch); negative when steepening
    :param slope_end:      asymptotic slope floor (default -4)
    :returns:              extrapolated cross section in cm^2/g, shape (n,)
    """
    v_0 = v[0]
    log_x = np.log(v / v_0)
    expon = np.maximum(slope_at_v0 + slope_of_slope * log_x, slope_end)
    log_sigma = np.zeros(len(v))
    log_sigma[1:] = cumulative_trapezoid(expon, log_x)
    return sigma_v0 * np.exp(log_sigma)

def get_slope_curvature_at_vmatch(v, sigma_v):
    """
    Estimate the curvature of the logarithmic slope of the velocity dependent cross section
     d^2(log sigma)/d(log v)^2 at the last tabulated velocity

    :param v:       velocity array in km/s, shape (n_v,)
    :param sigma_v: cross section strength shape (n_v,)
    :returns:       scalar d^2(log sigma)/d(log v)^2 evaluated at v[-1]
    """
    logv = np.log(v)
    slope = np.gradient(np.log(sigma_v), logv)
    return np.gradient(slope, logv)[-1]

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


_TABLES = {}
def get_phases(log10_mass_ratio, log10_alpha, potential='REPULSIVE_YUKAWA'):
    """
    Return the velocity grid and phase shifts for a given parameter combination
    (log10 v in km/s, and delta_ell(v) for ell = 0 .. N_ell-1). Each table is read
    from disk once and cached.
    """
    if potential not in _TABLES:
        with np.load(files("pykawa.data").joinpath(f"{potential}_phases.npz")) as d:
            _TABLES[potential] = {k: d[k] for k in d.files}
    data = _TABLES[potential]

    scale = 10 if potential == 'REPULSIVE_YUKAWA' else 100
    mphi_key  = int(round(log10_mass_ratio * scale))
    alpha_key = int(round(-log10_alpha * scale))  # stored as positive integers
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

    return data["v"][i, j].astype(np.float64), data["phases"][i, j].astype(np.float64)