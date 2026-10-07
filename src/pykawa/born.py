"""
Analytic Born-approximation cross sections for a Yukawa potential.

These are exact perturbative results, valid when the Born parameter

    beta = 2 alpha_chi m_phi / (m_chi v^2)

is small. Since beta scales as 1/v^2, the Born approximation becomes increasingly accurate at high
relative velocity, which makes it a cheap replacement for partial wave summation there -- exactly the
regime where the number of contributing partial waves makes the exact calculation expensive.

All results are independent of the sign of the potential: the amplitude enters squared, so Born cannot
distinguish attractive from repulsive scattering.
"""
import numpy as np

C_KMS = 299792.458            # speed of light [km / s]
GEV_INV2_TO_CM2 = 0.3894e-27  # 1 GeV^-2 in cm^2
GEV_TO_GRAM = 1.783e-24       # 1 GeV/c^2 in grams

_SERIES_THRESHOLD = 0.1  # below this R the closed forms lose precision to cancellation


def momentum_ratio(v_kms, m_chi, m_phi):
    """
    Calculate R = m_chi v / (2 m_phi), the ratio of the momentum in the centre of mass frame to the
    mediator mass. Note that some references instead define this quantity as m_chi v / m_phi, which
    is twice this value.
    :param v_kms: relative velocity [km / s]
    :param m_chi: dark matter mass [GeV]
    :param m_phi: mediator mass [GeV]
    :return: R (dimensionless)
    """
    return m_chi * (np.asarray(v_kms, dtype=float) / C_KMS) / (2 * m_phi)


def born_parameter(v_kms, m_chi, m_phi, alpha_chi):
    """
    Calculate beta = 2 alpha_chi m_phi / (m_chi v^2), the expansion parameter of the Born series. The
    approximation is accurate to roughly 1% for beta < 0.01, to tens of percent by beta ~ 0.1, and
    fails for beta > 1.
    :param v_kms: relative velocity [km / s]
    :param m_chi: dark matter mass [GeV]
    :param m_phi: mediator mass [GeV]
    :param alpha_chi: the coupling constant
    :return: beta (dimensionless)
    """
    v = np.asarray(v_kms, dtype=float) / C_KMS
    return 2 * alpha_chi * m_phi / (m_chi * v ** 2)


def _bracket_total(R):
    """
    Angular integral of the Born differential cross section with unit weight
    :param R: m_chi v / (2 m_phi)
    :return: F(R) for the total cross section
    """
    R = np.asarray(R, dtype=float)
    return 8 * R ** 4 / (1 + 4 * R ** 2)


def _bracket_momentum(R):
    """
    Angular integral of the Born differential cross section weighted by (1 - cos(theta))
    :param R: m_chi v / (2 m_phi)
    :return: F(R) for the momentum transfer cross section
    """
    R = np.asarray(R, dtype=float)
    out = np.empty_like(R)
    small = R < _SERIES_THRESHOLD
    R_small, R_large = R[small], R[~small]
    out[small] = 8 * R_small ** 4 - (128 / 3) * R_small ** 6 + 192 * R_small ** 8
    z = 4 * R_large ** 2
    out[~small] = np.log(1 + z) - z / (1 + z)
    return out


def _bracket_viscosity(R):
    """
    Angular integral of the Born differential cross section weighted by sin^2(theta)
    :param R: m_chi v / (2 m_phi)
    :return: F(R) for the viscosity cross section
    """
    R = np.asarray(R, dtype=float)
    out = np.empty_like(R)
    small = R < _SERIES_THRESHOLD
    R_small, R_large = R[small], R[~small]
    out[small] = (16 / 3) * R_small ** 4 - (64 / 3) * R_small ** 6 + (384 / 5) * R_small ** 8
    out[~small] = 2 * ((1 + 1 / (2 * R_large ** 2)) * np.log(1 + 4 * R_large ** 2) - 2)
    return out


_BRACKETS = {'TOTAL': _bracket_total,
             'MOMENTUM': _bracket_momentum,
             'VISCOSITY': _bracket_viscosity}


def sigma_born(v_kms, m_chi, m_phi, alpha_chi, cross_section_type='VISCOSITY', per_mass=True):
    """
    Calculate a Born-approximation cross section for a Yukawa potential

        sigma = (8 pi alpha_chi^2 / (m_chi^2 v^4)) * F(R),     R = m_chi v / (2 m_phi)

    with F given by the angular integral appropriate to the requested cross section:

        TOTAL:      F = 8 R^4 / (1 + 4 R^2)
        MOMENTUM:   F = ln(1 + 4R^2) - 4R^2 / (1 + 4R^2)
        VISCOSITY:  F = 2 [(1 + 1/(2R^2)) ln(1 + 4R^2) - 2]

    Below R = 0.1 the MOMENTUM and VISCOSITY forms are evaluated from their series expansions; the
    closed forms are differences of nearly equal numbers there and lose all precision, returning
    negative values for R < 0.01.
    :param v_kms: relative velocity [km / s]
    :param m_chi: dark matter mass [GeV]
    :param m_phi: mediator mass [GeV]
    :param alpha_chi: the coupling constant
    :param cross_section_type: one of 'VISCOSITY', 'MOMENTUM', or 'TOTAL'
    :param per_mass: if True, return sigma / m_chi in cm^2 / g; if False, return sigma in GeV^-2
    :return: the cross section, same shape as v_kms
    """
    if cross_section_type not in _BRACKETS:
        raise Exception('cross_section_type must be one of ' + str(sorted(_BRACKETS.keys())) +
                        ', got ' + str(cross_section_type))
    v = np.asarray(v_kms, dtype=float) / C_KMS
    R = m_chi * v / (2 * m_phi)
    sigma = 8 * np.pi * alpha_chi ** 2 / (m_chi ** 2 * v ** 4) * _BRACKETS[cross_section_type](R)
    if not per_mass:
        return sigma
    return (sigma / m_chi) * GEV_INV2_TO_CM2 / GEV_TO_GRAM


def sigma_viscosity_born(v_kms, m_chi, m_phi, alpha_chi, per_mass=True):
    """
    Calculate the Born viscosity cross section,
    sigma_V = integral of (dsigma/dOmega) sin^2(theta) dOmega
    :param v_kms: relative velocity [km / s]
    :param m_chi: dark matter mass [GeV]
    :param m_phi: mediator mass [GeV]
    :param alpha_chi: the coupling constant
    :param per_mass: if True, return sigma / m_chi in cm^2 / g; if False, return sigma in GeV^-2
    :return: the viscosity cross section
    """
    return sigma_born(v_kms, m_chi, m_phi, alpha_chi, 'VISCOSITY', per_mass)


def sigma_momentum_born(v_kms, m_chi, m_phi, alpha_chi, per_mass=True):
    """
    Calculate the Born momentum transfer cross section,
    sigma_T = integral of (dsigma/dOmega) (1 - cos(theta)) dOmega
    :param v_kms: relative velocity [km / s]
    :param m_chi: dark matter mass [GeV]
    :param m_phi: mediator mass [GeV]
    :param alpha_chi: the coupling constant
    :param per_mass: if True, return sigma / m_chi in cm^2 / g; if False, return sigma in GeV^-2
    :return: the momentum transfer cross section
    """
    return sigma_born(v_kms, m_chi, m_phi, alpha_chi, 'MOMENTUM', per_mass)


def sigma_total_born(v_kms, m_chi, m_phi, alpha_chi, per_mass=True):
    """
    Calculate the Born total cross section, sigma = integral of (dsigma/dOmega) dOmega
    :param v_kms: relative velocity [km / s]
    :param m_chi: dark matter mass [GeV]
    :param m_phi: mediator mass [GeV]
    :param alpha_chi: the coupling constant
    :param per_mass: if True, return sigma / m_chi in cm^2 / g; if False, return sigma in GeV^-2
    :return: the total cross section
    """
    return sigma_born(v_kms, m_chi, m_phi, alpha_chi, 'TOTAL', per_mass)
