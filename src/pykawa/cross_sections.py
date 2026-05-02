import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from joblib import Parallel, delayed
from tqdm import tqdm
from pykawa.utils import get_slope_at_vmatch, extrapolate
from pykawa.partial_wave_sums import partial_wave_sum_momentum, partial_wave_sum_viscosity, partial_wave_sum_total


class CrossSection(object):
    """
    Main class for an SIDM cross section. The class is intialized from a set of velocities (v) and a
    cross section strength sigma(v). It can also be initialized from a file that contains these quantities

    """
    def __init__(self, v, sigma_v, v_power=4.0, log10alpha=None, log10_mass_ratio=None,
                 interp_method='quadratic', phase_shifts=None):
        """
        log10_mass_ratio= log10(m_chi / m_phi)
        """
        vmax = v[-1]
        self.log10alpha = log10alpha
        self.log10_mass_ratio = log10_mass_ratio
        self.phase_shifts = phase_shifts
        # at low v
        v_extension_low = np.logspace(-4, np.log10(v[0]), 32)[:-1]
        v_extension_high = np.logspace(np.log10(vmax), 4.0, 32)
        sigma_v_extension_low = sigma_v[0] * np.ones_like(v_extension_low)
        #sigma_v_extension_high = sigma_v[-1] * (v_extension_high / vmax) ** -v_power
        slope_at_v0 = get_slope_at_vmatch(v, sigma_v)
        sigma_v_extension_high = extrapolate(v_extension_high,
                                             sigma_v[-1],
                                             slope_at_v0,
                                             slope_at_v0,
                                             -1*v_power)
        velocity = np.append(np.append(v_extension_low, v[0:-1]), v_extension_high)
        sigma = np.append(sigma_v_extension_low, np.append(sigma_v[0:-1], sigma_v_extension_high))

        self._velocity = velocity
        self.v = v
        self.sigma_v = sigma_v
        self.interp = interp1d(np.log(velocity),
                               np.log(sigma),
                               kind=interp_method)

    def particle_physics_params(self, amp_at_vref, v_ref=35):
        """
        Compute particle physics parameters for a cross-section.
        """
        xnorm = (np.log(v_ref))
        amp_at_vref_1 = np.exp(self.interp(xnorm))
        m_chi = (amp_at_vref / amp_at_vref_1) ** (-1 / 3)
        m_phi = m_chi / 10 ** self.log10_mass_ratio
        alpha_chi = 10 ** self.log10alpha
        params = {'m_chi': m_chi,  # GeV
                  'm_phi': m_phi * 1000,  # to MeV
                  'alpha_chi': alpha_chi}
        return params

    @classmethod
    def from_phase_shifts(cls, base_path,
                          log10alpha,
                          log10_mass_ratio,
                          cross_section_type='VISCOSITY',
                          v_power=4.0,
                          num_decimal=2):
        """
        Load the velocity and sigma(v) from a file
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
        log10v, _, _phase_shifts = data[:, 0], data[:, 1], data[:, 2]

        if cross_section_type == 'VISCOSITY':
            sigma_v = partial_wave_sum_viscosity(10 ** log10v,
                                                 phase_shifts)
            log10sigma = np.log10(sigma_v)

        elif cross_section_type == 'MOMENTUM':
            sigma_v = partial_wave_sum_momentum(10 ** log10v,
                                                 phase_shifts)
            log10sigma = np.log10(sigma_v)

        elif cross_section_type == 'TOTAL':
            sigma_v = partial_wave_sum_total(10 ** log10v,
                                                 phase_shifts)
            log10sigma = np.log10(sigma_v)

        return CrossSection(10 ** log10v,
                            10 ** log10sigma,
                            v_power,
                            log10alpha,
                            log10_mass_ratio,
                            phase_shifts=phase_shifts)


    @classmethod
    def from_file(cls, base_path, log10alpha, log10_mass_ratio, v_power=4.0, num_decimal=2):
        """
        Load the velocity and sigma(v) from a file
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
        log10v, log10sigma = data[:, 0], data[:, 1]
        return CrossSection(10 ** log10v,
                            10 ** log10sigma,
                            v_power,
                            log10alpha,
                            log10_mass_ratio)

    def thermal_average(self, v0, amp_at_vref, v_ref=35):
        """
        compute the thermally averaged cross section v^5 * sigma(v) d^3
        """
        _v = np.logspace(np.log10(self._velocity[0]), np.log10(self._velocity[-1]), 1000)
        _sigmav = self._evaluate_cross_section(_v, amp_at_vref, v_ref)
        kernel = _v ** 7 * np.exp(-0.25 * _v ** 2 / v0 ** 2)
        return np.trapz(kernel * _sigmav, _v) / np.trapz(kernel, _v)

    def thermal_average_integrand(self, v0, amp_at_vref, v_ref=35):
        """
        integrand for thermal average
        """
        _v = np.logspace(np.log10(self._velocity[0]), np.log10(self._velocity[-1]), 1000)
        _sigmav = self._evaluate_cross_section(_v, amp_at_vref, v_ref)
        kernel = _v ** 7 * np.exp(-0.25 * _v ** 2 / v0 ** 2)
        return _v, kernel * _sigmav

    def _evaluate_cross_section(self, v, amp_at_vref, v_ref=35):
        """
        evaluate the cross section at a particular v, such that sigma(v_ref) = amp_at_vref
        """
        if isinstance(v, list) or isinstance(v, np.ndarray):
            v_eval = np.ones_like(v) * v
            inds_low = np.where(v < self._velocity[0])[0]
            v_eval[inds_low] = self._velocity[0]
            inds_high = np.where(v > self._velocity[-1])[0]
            v_eval[inds_high] = self._velocity[-1]
        else:
            v_eval = max(v, self._velocity[0])
            v_eval = min(v_eval, self._velocity[-1])
        sigma = np.exp(self.interp(np.log(v_eval)))
        if amp_at_vref is None:
            pass
        else:
            norm = np.exp(self.interp(np.log(v_ref)))
            sigma *= amp_at_vref / norm
        return sigma

    def __call__(self, v, amp_at_vref, v_ref=35):

        return self._evaluate_cross_section(v, amp_at_vref, v_ref)

class CrossSectionInterpolator(object):
    """
    This class interpolates a series of CrossSection classes to allow a continuous representation of the parameter space
    """
    def __init__(self, v, log10alpha_values, log10_mass_ratio_values, cross_section_table,
                 interp_method='linear', v_ref=35, thermal_average_interpolation_class=None):

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

        cross_section_table = np.empty((len(log10alpha_values), len(log10_mass_ratio_values), len(v)))
        for i, log10alpha in enumerate(log10alpha_values):
            for j, log10_mass_ratio in enumerate(log10_mass_ratio_values):
                cross = CrossSection.from_file(base_path, log10_mass_ratio, log10alpha, v_power,
                                               filename_power=filename_power)
                cross_section_table[i, j, :] = cross(v, None)
        return CrossSectionInterpolator(v, log10alpha_values, log10_mass_ratio_values, cross_section_table)

    def setup_thermal_average_interp(self, log10v0_grid, nsteps=20):

        if self._thermal_average_interpolation_class is not None:
            print('Overriding current thermal average interpolation class... ')

        amp_at_vref_ref = 1.0
        log10alpha_min = self.log10alpha_values[0]
        log10alpha_max = self.log10alpha_values[-1]
        log10_mass_ratio_min = self.log10_mass_ratio_values[0]
        log10_mass_ratio_max = self.log10_mass_ratio_values[-1]
        nsteps_v0 = len(log10v0_grid)

        # Build the 3D grid axes (amp_at_vref dropped)
        log10alpha_grid = np.linspace(log10alpha_min, log10alpha_max, nsteps)
        log10_mass_ratio_grid = np.linspace(log10_mass_ratio_min, log10_mass_ratio_max, nsteps)

        # Flatten all grid points
        indices = [
            (i, j, k, lv0, lalpha, lmr)
            for i, lv0 in enumerate(log10v0_grid)
            for j, lalpha in enumerate(log10alpha_grid)
            for k, lmr in enumerate(log10_mass_ratio_grid)
        ]

        def _eval(i, j, k, lv0, lalpha, lmr):
            val = self.thermal_average(
                v0=10 ** lv0,
                amp_at_vref=amp_at_vref_ref,
                log10alpha=lalpha,
                log10_mass_ratio=lmr,
                use_interp=False
            )
            return (i, j, k, val)

        # Run in parallel
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_eval)(*args) for args in tqdm(
                indices,
                total=nsteps_v0 * nsteps ** 2,
                desc="Building interpolation grid",
                unit="eval",
            )
        )

        # Reassemble into the 3D array
        values = np.empty((nsteps_v0, nsteps, nsteps))
        for i, j, k, val in results:
            values[i, j, k] = val

        log10_values = np.log10(values)
        points = (log10v0_grid, log10alpha_grid, log10_mass_ratio_grid)

        self._thermal_average_interpolation_class = RegularGridInterpolator(
            points,
            log10_values,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )

        return points, log10_values, self._thermal_average_interpolation_class

    def _thermal_average_interpolation(self, v0, amp_at_vref, log10alpha, log10_mass_ratio):

        if self._thermal_average_interpolation_class is None:
            raise Exception('interpolation method for the thermal average not loaded with the class. '
                            'Should run the thermal average function with use_interp=False.')
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

        if use_interp:
            return self._thermal_average_interpolation(v0,
                                                       amp_at_vref,
                                                       log10alpha,
                                                       log10_mass_ratio)
        else:
            _v = np.logspace(np.log10(0.05 * v0), np.log10(150 * v0), num_steps)
            _sigmav = self._evaluate_cross_section(_v, amp_at_vref, log10alpha, log10_mass_ratio)
            kernel = _v ** 7 * np.exp(-0.25 * _v ** 2 / v0 ** 2)
            return np.trapz(kernel * _sigmav, _v) / np.trapz(kernel, _v)

    def thermal_average_integrand(self, v0, amp_at_vref, log10alpha, log10_mass_ratio):

        _v = np.logspace(np.log10(0.01 * v0), np.log10(100 * v0), 2000)
        _sigmav = self._evaluate_cross_section(_v, amp_at_vref, log10alpha, log10_mass_ratio)
        kernel = _v ** 7 * np.exp(-0.25 * _v ** 2 / v0 ** 2)
        return _v, kernel * _sigmav

    def particle_physics_params(self, log10alpha, log10_mass_ratio, amp_at_vref, v_ref=35):

        xnorm = (log10alpha, log10_mass_ratio, np.log(v_ref))
        amp_at_vref_1 = np.exp(self.interp(xnorm))
        m_chi = (amp_at_vref / amp_at_vref_1) ** (-1 / 3)
        m_phi = m_chi / 10 ** log10_mass_ratio
        alpha_chi = 10 ** log10alpha
        params = {'m_chi': m_chi,  # GeV
                  'm_phi': m_phi * 1000,  # to MeV
                  'alpha_chi': alpha_chi}
        return params

    def _evaluate_cross_section(self, v, amp_at_vref, log10alpha, log10_mass_ratio):

        if isinstance(v, list) or isinstance(v, np.ndarray):
            v_eval = np.ones_like(v) * v
            inds_low = np.where(v < self._v[0])[0]
            v_eval[inds_low] = self._v[0]
            inds_high = np.where(v > self._v[-1])[0]
            v_eval[inds_high] = self._v[-1]
        else:
            v_eval = max(v, self._v[0])
            v_eval = min(v_eval, self._v[-1])
        x = (log10alpha, log10_mass_ratio, np.log(v_eval))
        sigma = np.exp(self.interp(x))
        if amp_at_vref is None:
            pass
        else:
            xnorm = (log10alpha, log10_mass_ratio, np.log(self._vref))
            norm = np.exp(self.interp(xnorm))
            sigma *= amp_at_vref / norm
        return sigma

    def __call__(self, v, amp_at_vref, log10alpha, log10_mass_ratio):

        return self._evaluate_cross_section(v, amp_at_vref, log10alpha, log10_mass_ratio)

