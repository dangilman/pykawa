import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from joblib import Parallel, delayed
from tqdm import tqdm
from pykawa.utils import get_slope_at_vmatch, extrapolate_cross_section
from pykawa.partial_wave_sums import (partial_wave_sum_momentum,
                                      partial_wave_sum_viscosity,
                                      partial_wave_sum_total,
                                      partial_wave_sum_angular)


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
        sigma_v_extension_high = extrapolate_cross_section(v_extension_high,
                                             sigma_v[-1],
                                             slope_at_v0,
                                             slope_at_v0,
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
            weights /= np.trapz(weights, theta_grid, axis=1)[:, np.newaxis]
            cdfs = np.cumsum(weights, axis=1) * (theta_grid[1] - theta_grid[0])
            cdfs /= cdfs[:, -1:]

            self._scattering_angle_theta_grid = theta_grid
            self._scattering_angle_cdfs = cdfs  # (n_v, n_theta)
        return self._scattering_angle_cdfs, self._scattering_angle_theta_grid

    def sample_scattering_angle(self, v, n_samples=1):
        """
        Sample scattering angles θ from the distribution dσ/dΩ sinθ at a given velocity,
        using inverse CDF sampling on precomputed CDFs. The CDF at the requested velocity
        is obtained by log-linear interpolation between the two nearest tabulated velocities.
        Calls setup_scattering_angle_sampler() automatically on first use.

        :param v:         velocity in km/s (scalar); must be within the range of self.v
        :param n_samples: number of angles to sample
        :returns:         array of sampled scattering angles in radians, shape (n_samples,)
        """
        scattering_angle_cdfs, scattering_angle_theta_grid = self.setup_scattering_angle_sampler()
        log10v_data = np.log10(self.v)
        log10v_query = np.log10(v)
        idx = np.searchsorted(log10v_data, log10v_query)
        idx = np.clip(idx, 1, len(log10v_data) - 1)
        lo, hi = idx - 1, idx
        t = (log10v_query - log10v_data[lo]) / (log10v_data[hi] - log10v_data[lo])
        cdf = (1 - t) * scattering_angle_cdfs[lo] + t * scattering_angle_cdfs[hi]

        u = np.random.uniform(0, 1, n_samples)
        return np.interp(u, cdf, self._scattering_angle_theta_grid)

    def particle_physics_params(self, amp_at_vref, v_ref=35):
        """
        Infer the particle physics parameters (m_χ, m_φ, α_χ) that produce a cross section
        of amp_at_vref cm²/g at velocity v_ref km/s.

        :param amp_at_vref: desired cross section amplitude in cm²/g at v_ref
        :param v_ref:       reference velocity in km/s (default 35)
        :returns:           dict with keys 'm_chi' (GeV), 'm_phi' (MeV), 'alpha_chi'
        """
        xnorm = np.log(v_ref)
        amp_at_vref_1 = np.exp(self.interp(xnorm))
        m_chi = (amp_at_vref / amp_at_vref_1) ** (-1 / 3)
        m_phi = m_chi / 10 ** self.log10_mass_ratio
        alpha_chi = 10 ** self.log10alpha
        params = {'m_chi': m_chi,       # GeV
                  'm_phi': m_phi * 1000,  # MeV
                  'alpha_chi': alpha_chi}
        return params

    def thermal_average(self, v0, amp_at_vref, v_ref=35):
        """
        Compute the thermally averaged cross section ⟨σv⟩ for a Maxwell-Boltzmann
        distribution with velocity dispersion v0:
            ⟨σ⟩ = ∫ σ(v) v^7 exp(-v²/4v0²) dv / ∫ v^7 exp(-v²/4v0²) dv

        :param v0:          velocity dispersion in km/s
        :param amp_at_vref: cross section amplitude in cm²/g at v_ref
        :param v_ref:       reference velocity in km/s (default 35)
        :returns:           thermally averaged cross section in cm²/g
        """
        _v = np.logspace(np.log10(self._velocity[0]), np.log10(self._velocity[-1]), 1000)
        _sigmav = self._evaluate_cross_section(_v, amp_at_vref, v_ref)
        kernel = _v ** 7 * np.exp(-0.25 * _v ** 2 / v0 ** 2)
        return np.trapz(kernel * _sigmav, _v) / np.trapz(kernel, _v)

    def thermal_average_integrand(self, v0, amp_at_vref, v_ref=35):
        """
        Return the integrand of the thermal average for diagnostic purposes.

        :param v0:          velocity dispersion in km/s
        :param amp_at_vref: cross section amplitude in cm²/g at v_ref
        :param v_ref:       reference velocity in km/s (default 35)
        :returns:           tuple (_v, integrand) where _v is the velocity array in km/s
                            and integrand = σ(v) * v^7 * exp(-v²/4v0²)
        """
        _v = np.logspace(np.log10(self._velocity[0]), np.log10(self._velocity[-1]), 1000)
        _sigmav = self._evaluate_cross_section(_v, amp_at_vref, v_ref)
        kernel = _v ** 7 * np.exp(-0.25 * _v ** 2 / v0 ** 2)
        return _v, kernel * _sigmav

    def _evaluate_cross_section(self, v, amp_at_vref, v_ref=35):
        """
        Evaluate the interpolated cross section at velocity v, normalized so that
        σ(v_ref) = amp_at_vref. Velocities outside the tabulated range are clamped
        to the nearest boundary value before interpolation.

        :param v:           velocity in km/s, scalar or array
        :param amp_at_vref: desired cross section in cm²/g at v_ref; pass None to
                            return the unnormalized interpolated values
        :param v_ref:       reference velocity in km/s (default 35)
        :returns:           cross section in cm²/g, same shape as v
        """
        if isinstance(v, (list, np.ndarray)):
            v_eval = np.ones_like(v) * v
            v_eval[np.where(v < self._velocity[0])[0]] = self._velocity[0]
            v_eval[np.where(v > self._velocity[-1])[0]] = self._velocity[-1]
        else:
            v_eval = max(v, self._velocity[0])
            v_eval = min(v_eval, self._velocity[-1])
        sigma = np.exp(self.interp(np.log(v_eval)))
        if amp_at_vref is not None:
            norm = np.exp(self.interp(np.log(v_ref)))
            sigma *= amp_at_vref / norm
        return sigma

    def __call__(self, v, amp_at_vref, v_ref=35):
        """
        Evaluate the cross section at velocity v. Convenience wrapper for
        _evaluate_cross_section.

        :param v:           velocity in km/s, scalar or array
        :param amp_at_vref: desired cross section in cm²/g at v_ref
        :param v_ref:       reference velocity in km/s (default 35)
        :returns:           cross section in cm²/g, same shape as v
        """
        return self._evaluate_cross_section(v, amp_at_vref, v_ref)


class CrossSectionInterpolator(object):
    """
    Interpolates over a grid of CrossSection instances to give a continuous
    representation of σ(v; log10α, log10(m_χ/m_φ)). Useful for parameter
    inference where the cross section must be evaluated at many parameter combinations.
    """

    def __init__(self, v, log10alpha_values, log10_mass_ratio_values, cross_section_table,
                 interp_method='linear', v_ref=35, thermal_average_interpolation_class=None):
        """
        :param v:                       velocities in km/s, shape (n_v,)
        :param log10alpha_values:        grid of log10α values, shape (n_alpha,)
        :param log10_mass_ratio_values:  grid of log10(m_χ/m_φ) values, shape (n_mr,)
        :param cross_section_table:      cross section values, shape (n_alpha, n_mr, n_v)
        :param interp_method:            interpolation method for RegularGridInterpolator
        :param v_ref:                    reference velocity in km/s for normalization
        :param thermal_average_interpolation_class: optional precomputed thermal average
                                         interpolator; set via setup_thermal_average_interp()
        """
        self._vref = v_ref
        self._v = v
        self.log10alpha_values = log10alpha_values
        self.log10_mass_ratio_values = log10_mass_ratio_values
        points = (log10alpha_values, log10_mass_ratio_values, np.log(v))
        self.interp = RegularGridInterpolator(points,
                                              np.log(cross_section_table),
                                              method=interp_method)
        self._thermal_average_interpolation_class = thermal_average_interpolation_class

    @classmethod
    def from_file(cls, base_path, v, log10alpha_values, log10_mass_ratio_values, v_power=4.0,
                  filename_power=10):
        """
        Build a CrossSectionInterpolator by loading a grid of phase shift files and
        evaluating each cross section on the velocity array v.

        :param base_path:                path to directory containing phase shift files
        :param v:                        velocities in km/s at which to evaluate σ, shape (n_v,)
        :param log10alpha_values:        grid of log10α values, shape (n_alpha,)
        :param log10_mass_ratio_values:  grid of log10(m_χ/m_φ) values, shape (n_mr,)
        :param v_power:                  power law index for high-velocity extrapolation
        :param filename_power:           multiplier used in filename encoding
        :returns:                        CrossSectionInterpolator instance
        """
        cross_section_table = np.empty((len(log10alpha_values), len(log10_mass_ratio_values), len(v)))
        for i, log10alpha in enumerate(log10alpha_values):
            for j, log10_mass_ratio in enumerate(log10_mass_ratio_values):
                cross = CrossSection.from_file(base_path, log10_mass_ratio, log10alpha, v_power,
                                               filename_power=filename_power)
                cross_section_table[i, j, :] = cross(v, None)
        return CrossSectionInterpolator(v, log10alpha_values, log10_mass_ratio_values, cross_section_table)

    def setup_thermal_average_interp(self, log10v0_grid, nsteps=20):
        """
        Precompute a 3D interpolation table for the thermal average ⟨σ⟩(v0; log10α, log10(m_χ/m_φ))
        on a regular grid, using parallel evaluation. After calling this method,
        thermal_average() will use the interpolator instead of direct integration.

        :param log10v0_grid: 1D array of log10(v0) values defining the v0 grid
        :param nsteps:       number of grid points along each of the log10α and
                             log10(m_χ/m_φ) axes (default 20)
        :returns:            tuple (points, log10_values, interpolator)
        """
        if self._thermal_average_interpolation_class is not None:
            print('Overriding current thermal average interpolation class... ')

        amp_at_vref_ref = 1.0
        nsteps_v0 = len(log10v0_grid)
        log10alpha_grid = np.linspace(self.log10alpha_values[0], self.log10alpha_values[-1], nsteps)
        log10_mass_ratio_grid = np.linspace(self.log10_mass_ratio_values[0],
                                            self.log10_mass_ratio_values[-1], nsteps)

        indices = [
            (i, j, k, lv0, lalpha, lmr)
            for i, lv0 in enumerate(log10v0_grid)
            for j, lalpha in enumerate(log10alpha_grid)
            for k, lmr in enumerate(log10_mass_ratio_grid)
        ]

        def _eval(i, j, k, lv0, lalpha, lmr):
            val = self.thermal_average(v0=10 ** lv0, amp_at_vref=amp_at_vref_ref,
                                       log10alpha=lalpha, log10_mass_ratio=lmr,
                                       use_interp=False)
            return (i, j, k, val)

        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_eval)(*args) for args in tqdm(
                indices, total=nsteps_v0 * nsteps ** 2,
                desc="Building interpolation grid", unit="eval"))

        values = np.empty((nsteps_v0, nsteps, nsteps))
        for i, j, k, val in results:
            values[i, j, k] = val

        log10_values = np.log10(values)
        points = (log10v0_grid, log10alpha_grid, log10_mass_ratio_grid)
        self._thermal_average_interpolation_class = RegularGridInterpolator(
            points, log10_values, method='linear', bounds_error=False, fill_value=None)

        return points, log10_values, self._thermal_average_interpolation_class

    def _thermal_average_interpolation(self, v0, amp_at_vref, log10alpha, log10_mass_ratio):
        """
        Evaluate the precomputed thermal average interpolation table.
        Raises an exception if setup_thermal_average_interp() has not been called.

        :param v0:               velocity dispersion in km/s
        :param amp_at_vref:      cross section amplitude in cm²/g at v_ref
        :param log10alpha:       log10 of the coupling constant α
        :param log10_mass_ratio: log10 of the mass ratio m_χ / m_φ
        :returns:                thermally averaged cross section in cm²/g
        """
        if self._thermal_average_interpolation_class is None:
            raise Exception('Thermal average interpolator not set up. '
                            'Call setup_thermal_average_interp() first.')
        v0 = np.atleast_1d(v0)
        log10alpha = np.atleast_1d(log10alpha)
        log10_mass_ratio = np.atleast_1d(log10_mass_ratio)
        amp_at_vref = np.atleast_1d(amp_at_vref)
        point = np.column_stack([np.log10(v0), log10alpha, log10_mass_ratio])
        log10_result = self._thermal_average_interpolation_class(point)
        result = 10 ** log10_result * amp_at_vref
        return float(result) if result.size == 1 else result

    def thermal_average(self, v0, amp_at_vref, log10alpha, log10_mass_ratio,
                        use_interp=True, num_steps=200):
        """
        Compute the thermally averaged cross section ⟨σ⟩ for a Maxwell-Boltzmann
        distribution with velocity dispersion v0. Can use either direct numerical
        integration or the precomputed interpolation table.

        :param v0:               velocity dispersion in km/s
        :param amp_at_vref:      cross section amplitude in cm²/g at v_ref
        :param log10alpha:       log10 of the coupling constant α
        :param log10_mass_ratio: log10 of the mass ratio m_χ / m_φ
        :param use_interp:       if True, use the precomputed interpolation table
                                 (requires setup_thermal_average_interp() to have been called);
                                 if False, perform direct numerical integration (slower)
        :param num_steps:        number of velocity integration points when use_interp=False
        :returns:                thermally averaged cross section in cm²/g
        """
        if use_interp:
            return self._thermal_average_interpolation(v0, amp_at_vref, log10alpha, log10_mass_ratio)
        else:
            _v = np.logspace(np.log10(0.05 * v0), np.log10(150 * v0), num_steps)
            _sigmav = self._evaluate_cross_section(_v, amp_at_vref, log10alpha, log10_mass_ratio)
            kernel = _v ** 7 * np.exp(-0.25 * _v ** 2 / v0 ** 2)
            return np.trapz(kernel * _sigmav, _v) / np.trapz(kernel, _v)

    def thermal_average_integrand(self, v0, amp_at_vref, log10alpha, log10_mass_ratio):
        """
        Return the integrand of the thermal average for diagnostic purposes.

        :param v0:               velocity dispersion in km/s
        :param amp_at_vref:      cross section amplitude in cm²/g at v_ref
        :param log10alpha:       log10 of the coupling constant α
        :param log10_mass_ratio: log10 of the mass ratio m_χ / m_φ
        :returns:                tuple (_v, integrand) where _v is the velocity array in km/s
                                 and integrand = σ(v) * v^7 * exp(-v²/4v0²)
        """
        _v = np.logspace(np.log10(0.01 * v0), np.log10(100 * v0), 2000)
        _sigmav = self._evaluate_cross_section(_v, amp_at_vref, log10alpha, log10_mass_ratio)
        kernel = _v ** 7 * np.exp(-0.25 * _v ** 2 / v0 ** 2)
        return _v, kernel * _sigmav

    def particle_physics_params(self, log10alpha, log10_mass_ratio, amp_at_vref, v_ref=35):
        """
        Infer the particle physics parameters (m_χ, m_φ, α_χ) that produce a cross section
        of amp_at_vref cm²/g at velocity v_ref km/s, for given log10α and log10(m_χ/m_φ).

        :param log10alpha:       log10 of the coupling constant α
        :param log10_mass_ratio: log10 of the mass ratio m_χ / m_φ
        :param amp_at_vref:      desired cross section in cm²/g at v_ref
        :param v_ref:            reference velocity in km/s (default 35)
        :returns:                dict with keys 'm_chi' (GeV), 'm_phi' (MeV), 'alpha_chi'
        """
        xnorm = (log10alpha, log10_mass_ratio, np.log(v_ref))
        amp_at_vref_1 = np.exp(self.interp(xnorm))
        m_chi = (amp_at_vref / amp_at_vref_1) ** (-1 / 3)
        m_phi = m_chi / 10 ** log10_mass_ratio
        alpha_chi = 10 ** log10alpha
        params = {'m_chi': m_chi,
                  'm_phi': m_phi * 1000,
                  'alpha_chi': alpha_chi}
        return params

    def _evaluate_cross_section(self, v, amp_at_vref, log10alpha, log10_mass_ratio):
        """
        Evaluate the interpolated cross section at velocity v for given particle physics
        parameters, normalized so that σ(v_ref) = amp_at_vref. Velocities outside the
        tabulated range are clamped to the nearest boundary value.

        :param v:                velocity in km/s, scalar or array
        :param amp_at_vref:      desired cross section in cm²/g at v_ref; pass None to
                                 return unnormalized values
        :param log10alpha:       log10 of the coupling constant α
        :param log10_mass_ratio: log10 of the mass ratio m_χ / m_φ
        :returns:                cross section in cm²/g, same shape as v
        """
        if isinstance(v, (list, np.ndarray)):
            v_eval = np.ones_like(v) * v
            v_eval[np.where(v < self._v[0])[0]] = self._v[0]
            v_eval[np.where(v > self._v[-1])[0]] = self._v[-1]
        else:
            v_eval = max(v, self._v[0])
            v_eval = min(v_eval, self._v[-1])
        x = (log10alpha, log10_mass_ratio, np.log(v_eval))
        sigma = np.exp(self.interp(x))
        if amp_at_vref is not None:
            xnorm = (log10alpha, log10_mass_ratio, np.log(self._vref))
            norm = np.exp(self.interp(xnorm))
            sigma *= amp_at_vref / norm
        return sigma

    def __call__(self, v, amp_at_vref, log10alpha, log10_mass_ratio):
        """
        Evaluate the cross section at velocity v. Convenience wrapper for
        _evaluate_cross_section.

        :param v:                velocity in km/s, scalar or array
        :param amp_at_vref:      desired cross section in cm²/g at v_ref
        :param log10alpha:       log10 of the coupling constant α
        :param log10_mass_ratio: log10 of the mass ratio m_χ / m_φ
        :returns:                cross section in cm²/g, same shape as v
        """
        return self._evaluate_cross_section(v, amp_at_vref, log10alpha, log10_mass_ratio)
