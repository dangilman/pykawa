from scipy.integrate import cumulative_trapezoid
import numpy as np


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
