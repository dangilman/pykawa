import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from pykawa.utils import (get_slope_at_vmatch, get_slope_curvature_at_vmatch,
                          extrapolate_cross_section, get_phases)
from pykawa.partial_wave_sums import (partial_wave_sum_momentum,
                                      partial_wave_sum_viscosity,
                                      partial_wave_sum_total,
                                      partial_wave_sum_angular)
from importlib.resources import files

class CrossSection(object):
    """
    Main class for an SIDM cross section computed from partial wave phase shifts.
    The class is initialized from a set of velocities (v) and a cross section σ(v),
    and constructs a smooth interpolation that can be evaluated at arbitrary velocities.
    At low velocities the cross section is held constant; at high velocities it is
    extrapolated as a power law v^{-v_power}.
    """

    def __init__(self, v, sigma_v, v_power=4.0, log10alpha=None, log10_mass_ratio=None,
                 interp_method='quadratic', phase_shifts=None):
        """
        :param v:                 velocities in km/s, shape (n_v,), must be sorted ascending
        :param sigma_v:           cross section in cm²/g, shape (n_v,)
        :param v_power:           power law index for high-velocity extrapolation; σ ∝ v^{-v_power}
        :param log10alpha:        log10 of the coupling constant α (stored as metadata)
        :param log10_mass_ratio:  log10 of the mass ratio m_χ / m_φ (stored as metadata)
        :param interp_method:     interpolation method passed to scipy interp1d (default 'quadratic')
        :param phase_shifts:      phase shift array, shape (n_v, lmax+1), stored for angle sampling
        """
        vmax = v[-1]
        self.log10alpha = log10alpha
        self.log10_mass_ratio = log10_mass_ratio
        self.phase_shifts = phase_shifts

        v_extension_low = np.logspace(-4, np.log10(v[0]), 32)[:-1]
        v_extension_high = np.logspace(np.log10(vmax), 4.0, 32)
        sigma_v_extension_low = sigma_v[0] * np.ones_like(v_extension_low)
        slope_at_v0 = get_slope_at_vmatch(v, sigma_v)
        curv_at_v0 = get_slope_curvature_at_vmatch(v, sigma_v)
        curv_at_v0 = min(curv_at_v0, -0.05)
        sigma_v_extension_high = extrapolate_cross_section(v_extension_high,
                                             sigma_v[-1],
                                             slope_at_v0,
                                             curv_at_v0,
                                             -1 * v_power)
        velocity = np.append(np.append(v_extension_low, v[0:-1]), v_extension_high)
        sigma = np.append(sigma_v_extension_low, np.append(sigma_v[0:-1], sigma_v_extension_high))

        self._velocity = velocity
        self.v = v
        self.sigma_v = sigma_v
        self.interp = interp1d(np.log(velocity),
                               np.log(sigma),
                               kind=interp_method)

    @classmethod
    def from_file(cls, base_path,
                  log10alpha,
                  log10_mass_ratio,
                  cross_section_type='VISCOSITY',
                  v_power=4.0,
                  num_decimal=2):
        """
        Initialize a CrossSection by loading phase shifts from a file and computing
        the integrated cross section via partial wave sums.
        The filename is constructed as:
            base_path + 'mphi{int(log10_mass_ratio * filename_power)}_alpha{int(-log10alpha * filename_power)}.txt'

        :param base_path:           path to the directory containing phase shift files
        :param log10alpha:          log10 of the coupling constant α
        :param log10_mass_ratio:    log10 of the mass ratio m_χ / m_φ
        :param cross_section_type:  one of 'VISCOSITY' (σ_V), 'MOMENTUM' (σ_T), or 'TOTAL' (σ_tot)
        :param v_power:             power law index for high-velocity extrapolation
        :param num_decimal:         number of decimal places encoded in the filename (2 or 3)
        :returns:                   CrossSection instance
        """
        if num_decimal == 2:
            filename_power = 100
        elif num_decimal == 3:
            filename_power = 1000
        else:
            raise ValueError("num_decimal must be 2 or 3")
        filename = (base_path +
                    'mphi' + str(int(round(filename_power * log10_mass_ratio))) +
                    '_alpha' + str(int(round(-filename_power * log10alpha))) +
                    '.txt')
        data = np.loadtxt(filename)
        log10v = data[:, 0]
        phase_shifts = data[:, 1:]
        return cls.from_phase_shifts(log10v,
                                     phase_shifts,
                                     cross_section_type,
                                     log10alpha,
                                     log10_mass_ratio,
                                     v_power)

    @classmethod
    def from_phase_shifts(cls, log10v, phase_shifts, cross_section_type,
                          log10alpha, log10_mass_ratio, v_power=4.0):
        """
        Initialize a CrossSection directly from arrays of log10 velocities and phase shifts,
        computing the integrated cross section via partial wave sums.

        :param log10v:              log10 of velocities in km/s, shape (n_v,)
        :param phase_shifts:        phase shift array, shape (n_v, lmax+1), zero-padded;
                                    each row is [δ_0, δ_1, ..., δ_lmax] for that velocity
        :param cross_section_type:  one of 'VISCOSITY' (σ_V), 'MOMENTUM' (σ_T), or 'TOTAL' (σ_tot)
        :param log10alpha:          log10 of the coupling constant α
        :param log10_mass_ratio:    log10 of the mass ratio m_χ / m_φ
        :param v_power:             power law index for high-velocity extrapolation
        :returns:                   CrossSection instance
        """
        if cross_section_type == 'VISCOSITY':
            sigma_v = partial_wave_sum_viscosity(10 ** log10v, phase_shifts)
        elif cross_section_type == 'MOMENTUM':
            sigma_v = partial_wave_sum_momentum(10 ** log10v, phase_shifts)
        elif cross_section_type == 'TOTAL':
            sigma_v = partial_wave_sum_total(10 ** log10v, phase_shifts)
        else:
            raise ValueError("cross_section_type must be 'VISCOSITY', 'MOMENTUM', or 'TOTAL'")

        return cls(10 ** log10v,
                   sigma_v,
                   v_power,
                   log10alpha,
                   log10_mass_ratio,
                   phase_shifts=phase_shifts)

    def setup_scattering_angle_sampler(self, n_theta=1000):
        """
        Precompute the CDFs of dσ/dΩ at each velocity in the phase shift data,
        for efficient repeated sampling of scattering angles via inverse CDF.
        The CDFs are cached after the first call; subsequent calls return immediately.
        Must be called (explicitly or via sample_scattering_angle) before sampling.

        :param n_theta: number of θ grid points used to construct the CDFs;
                        increase for better accuracy at high velocities where the
                        forward-scattering peak is sharp (default 1000)
        :returns:       tuple (cdfs, theta_grid) where cdfs has shape (n_v, n_theta)
                        and theta_grid has shape (n_theta,) in radians
        """
        if not hasattr(self, '_scattering_angle_cdfs'):
            if self.phase_shifts is None:
                raise ValueError("Phase shifts must be loaded to sample scattering angles. "
                                 "Initialize with from_phase_shifts().")

            theta_grid = np.linspace(0, np.pi, n_theta)
            dsigma = partial_wave_sum_angular(self.v, theta_grid, self.phase_shifts)

            weights = dsigma * np.sin(theta_grid)[np.newaxis, :]  # (n_v, n_theta)
            weights /= np.trapezoid(weights, theta_grid, axis=1)[:, np.newaxis]
            cdfs = np.cumsum(weights, axis=1) * (theta_grid[1] - theta_grid[0])
            cdfs /= cdfs[:, -1:]

            self._scattering_angle_theta_grid = theta_grid
            self._scattering_angle_cdfs = cdfs  # (n_v, n_theta)
        return self._scattering_angle_cdfs, self._scattering_angle_theta_grid

    def sample_scattering_angle(self, v, n_samples=1, n_theta=1000):
        """
        Sample scattering angles θ from the distribution dσ/dΩ sinθ at a given velocity,
        using inverse CDF sampling on precomputed CDFs. The CDF at the requested velocity
        is obtained by log-linear interpolation between the two nearest tabulated velocities.
        Calls setup_scattering_angle_sampler() automatically on first use.

        :param v:         velocity in km/s (scalar); must be within the range of self.v
        :param n_samples: number of angles to sample
        :param n_theta:   the resolution in angle on which to interpolate dσ/dΩ from 0 to pi
        :returns:         array of sampled scattering angles in radians, shape (n_samples,)
        """
        scattering_angle_cdfs, scattering_angle_theta_grid = self.setup_scattering_angle_sampler(
            n_theta
        )
        log10v_data = np.log10(self.v)
        log10v_query = np.log10(v)
        idx = np.searchsorted(log10v_data, log10v_query)
        idx = np.clip(idx, 1, len(log10v_data) - 1)
        lo, hi = idx - 1, idx
        t = (log10v_query - log10v_data[lo]) / (log10v_data[hi] - log10v_data[lo])
        cdf = (1 - t) * scattering_angle_cdfs[lo] + t * scattering_angle_cdfs[hi]

        u = np.random.uniform(0, 1, n_samples)
        return np.interp(u, cdf, self._scattering_angle_theta_grid)

    def convert_to_SIDM_params(self, amp_at_vref, v_ref=60):
        """
        Compute the m_chi, m_phi values corresponding to a set of alpha_chi, m_chi / m_phi, amp_at_vref values
        return: m_chi and m_phi in Gev
        """
        m_chi = self.compute_mchi(amp_at_vref, v_ref)
        m_phi = m_chi / 10 ** self.log10_mass_ratio
        return m_chi, m_phi

    def compute_mchi(self, amp_at_vref, v_ref=60):
        """
        Compute the DM particle mass m_chi for a given alpha, m_chi / m_phi, and amp_at_vref combination
        return m_chi in GeV
        """
        xnorm = np.log(v_ref)
        amp_at_vref_1 = np.exp(self.interp(xnorm))
        m_chi = np.exp((self.interp(xnorm) - np.log(amp_at_vref)) / 3)
        return m_chi

    def relic_density_log10_dsphr(self, v_ref=60, relic_coeff=0.1):
        """
        Compute the value of log10(sigma/m_chi) at v_ref [cm^2/g] that satisfies the
        thermal relic density constraint alpha_chi = relic_coeff * (m_chi / TeV).

        :param v_ref:       reference velocity in km/s
        :param relic_coeff: coefficient in alpha_chi = relic_coeff * (m_chi / TeV);
                            0.1 is the order-of-magnitude estimate for s-wave
                            annihilation into light mediators
        :returns:           log10(amp_at_vref) satisfying the relic constraint
        """
        # relic: alpha_chi = relic_coeff * (m_chi / TeV)  =>  m_chi [GeV] = 10^3 * alpha_chi / relic_coeff
        log10_m_chi = self.log10alpha - np.log10(relic_coeff) + 3.0
        # sigma/m ~ m_chi^{-3}, with the table normalized to m_chi = 1 GeV
        log10_amp_1 = float(self.interp(np.log(v_ref))) / np.log(10.0)
        return log10_amp_1 - 3.0 * log10_m_chi

    def particle_physics_params(self, amp_at_vref, v_ref=60):
        """
        Infer the particle physics parameters (m_χ, m_φ, α_χ) that produce a cross section
        of amp_at_vref cm²/g at velocity v_ref km/s.

        :param amp_at_vref: desired cross section amplitude in cm²/g at v_ref
        :param v_ref:       reference velocity in km/s (default 60)
        :returns:           dict with keys 'm_chi' (GeV), 'm_phi' (MeV), 'alpha_chi'
        """
        m_chi, m_phi = self.convert_to_SIDM_params(amp_at_vref, v_ref)
        alpha_chi = 10 ** self.log10alpha
        params = {'m_chi': m_chi,       # GeV
                  'm_phi': m_phi,       # GeV
                  'alpha_chi': alpha_chi}
        return params

    def thermal_average(self, v0, amp_at_vref, v_ref=60):
        """
        Compute the thermally averaged cross section ⟨σv⟩ for a Maxwell-Boltzmann
        distribution with velocity dispersion v0:
            ⟨σ⟩ = ∫ σ(v) v^7 exp(-v²/4v0²) dv / ∫ v^7 exp(-v²/4v0²) dv

        :param v0:          velocity dispersion in km/s
        :param amp_at_vref: cross section amplitude in cm²/g at v_ref
        :param v_ref:       reference velocity in km/s (default 60)
        :returns:           thermally averaged cross section in cm²/g
        """
        _v = np.logspace(np.log10(self._velocity[0]), np.log10(self._velocity[-1]), 1000)
        _sigmav = self._evaluate_cross_section(_v, amp_at_vref, v_ref)
        kernel = _v ** 7 * np.exp(-0.25 * _v ** 2 / v0 ** 2)
        return np.trapezoid(kernel * _sigmav, _v) / np.trapezoid(kernel, _v)

    def thermal_average_integrand(self, v0, amp_at_vref, v_ref=60):
        """
        Return the integrand of the thermal average for diagnostic purposes.

        :param v0:          velocity dispersion in km/s
        :param amp_at_vref: cross section amplitude in cm²/g at v_ref
        :param v_ref:       reference velocity in km/s (default 60)
        :returns:           tuple (_v, integrand) where _v is the velocity array in km/s
                            and integrand = σ(v) * v^7 * exp(-v²/4v0²)
        """
        _v = np.logspace(np.log10(self._velocity[0]), np.log10(self._velocity[-1]), 1000)
        _sigmav = self._evaluate_cross_section(_v, amp_at_vref, v_ref)
        kernel = _v ** 7 * np.exp(-0.25 * _v ** 2 / v0 ** 2)
        return _v, kernel * _sigmav

    def _evaluate_cross_section(self, v, amp_at_vref, v_ref=60):
        """
        Evaluate the interpolated cross section at velocity v, normalized so that
        σ(v_ref) = amp_at_vref. Velocities outside the tabulated range are clamped
        to the nearest boundary value before interpolation.

        :param v:           velocity in km/s, scalar or array
        :param amp_at_vref: desired cross section in cm²/g at v_ref; pass None to
                            return the unnormalized interpolated values
        :param v_ref:       reference velocity in km/s (default 60)
        :returns:           cross section in cm²/g, same shape as v
        """
        if isinstance(v, (list, np.ndarray)):
            v_eval = np.ones_like(v) * v
            v_eval[np.where(v < self._velocity[0])[0]] = self._velocity[0]
            v_eval[np.where(v > self._velocity[-1])[0]] = self._velocity[-1]
        else:
            v_eval = max(v, self._velocity[0])
            v_eval = min(v_eval, self._velocity[-1])
        if amp_at_vref is None:
            return np.exp(self.interp(np.log(v_eval)))
        return amp_at_vref * np.exp(self.interp(np.log(v_eval))
                                    - self.interp(np.log(v_ref)))

    def __call__(self, v, amp_at_vref, v_ref=60):
        """
        Evaluate the cross section at velocity v. Convenience wrapper for
        _evaluate_cross_section.

        :param v:           velocity in km/s, scalar or array
        :param amp_at_vref: desired cross section in cm²/g at v_ref
        :param v_ref:       reference velocity in km/s (default 60)
        :returns:           cross section in cm²/g, same shape as v
        """
        return self._evaluate_cross_section(v, amp_at_vref, v_ref)

