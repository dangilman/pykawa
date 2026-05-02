from scipy.integrate import cumulative_trapezoid
import numpy as np


def extrapolate(v, sigma_v0, slope_at_v0, slope_of_slope, slope_end=-4.0):
    """
    Extrapolate sigma(v) where d(log sigma)/d(log v) transitions linearly
    from a logarithmic slope equal to slope_at_v0 to slope end linearly with log10(v/v0)
    """
    v_0 = v[0]
    log_x = np.log(v / v_0)  # work in log space

    # Slope as a function of log(v/v0) — linear ramp that floors at slope_end
    expon = slope_at_v0 + slope_of_slope * log_x
    expon = np.maximum(expon, slope_end)

    # Integrate d(log sigma) = expon * d(log v) from 0 to log_x[i]
    log_sigma = np.zeros(len(v))
    log_sigma[1:] = cumulative_trapezoid(expon, log_x)
    return sigma_v0 * np.exp(log_sigma)

def get_slope_at_vmatch(v, sigma_v):
    dlogsigma_dlogv = np.gradient(np.log(sigma_v), np.log(v))
    return dlogsigma_dlogv[-1]
