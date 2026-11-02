from scipy.interpolate import RegularGridInterpolator
import numpy as np
from pykawa.cross_sections import CrossSection
from joblib import Parallel, delayed
from importlib.resources import files
from tqdm.auto import tqdm
from pykawa.yukawa_viscosity_even_odd import yukawa_viscosity_even_odd


class FermionChargedScalarCrossSection(object):
    """
    Viscosity cross section for chi-chibar scattering of symmetric fermionic dark matter with a charged
    scalar mediator
        sigma_V = (4 pi / k^2) [ 1/4 (S_even^att + S_odd^rep) + 3/4 (S_even^rep + S_odd^att) ]

    ATTRIBUTION: when using this class, please cite:
    - M. M. Flores, K. Petraki, and T. Pignard, Variable-phase methods and data for long-range scattering, arXiv:2610.30261.
    and also cite the dataset:
    - M. M. Flores, K. Petraki, and T. Pignard, YukawaVPS data: phase shifts, Sommerfeld factors,
    and cross sections for Yukawa scattering, Zenodo (2026), doi:10.5281/zenodo.23475703.
    """
    C_KMS = 299792.458
    GEV2_TO_CM2 = 0.389379e-27
    GEV_TO_G = 1.78266e-24

    def __init__(self, v_ref=10.0, log10alpha_range=(-5.2, -1.7),
                 log10_mass_ratio_range=(2.8, 5.4), cache_resolution=0.0025, pair_fraction=0.5,
                 viscosity_normalization=1.5):
        """
        :param v_ref: reference velocity at which the amplitude is pinned [km / s]
        :param log10alpha_range: validated range of log10 alpha_chi
        :param log10_mass_ratio_range: validated range of log10 m_chi / m_phi
        :param cache_resolution: grid spacing of the fixed velocity tables [dex]; None evaluates the
                                 package directly at every call, which is only practical for small arrays
        :param pair_fraction: fraction of particle pairs that are chi-chibar; 1/2 for symmetric dark matter,
                              1 to return sigma_chi-chibar itself
        :param viscosity_normalization: prefactor of the viscosity integral; 3/2 so that sigma_V equals the
                                        total cross section for isotropic scattering, 1 for the bare integral
        """
        if not 0.0 < pair_fraction <= 1.0:
            raise ValueError('pair_fraction must be in (0, 1], got %s' % pair_fraction)
        if not viscosity_normalization > 0.0:
            raise ValueError('viscosity_normalization must be positive, got %s' % viscosity_normalization)
        self._even_odd = yukawa_viscosity_even_odd
        self._pair_fraction = float(pair_fraction)
        self._viscosity_normalization = float(viscosity_normalization)
        self._vref = float(v_ref)
        self.log10alpha_values = np.array(log10alpha_range, dtype=float)
        self.log10_mass_ratio_values = np.array(log10_mass_ratio_range, dtype=float)
        self._cache_resolution = cache_resolution
        self._fixed_velocity_tables = {}
        self._thermal_average_tables = {}

    @property
    def v_ref(self):
        """
        Reference velocity in km/s
        """
        return self._vref

    @property
    def pair_fraction(self):
        """
        Fraction of particle pairs that are chi-chibar; every returned cross section is this times sigma_chi-chibar
        """
        return self._pair_fraction

    @property
    def viscosity_normalization(self):
        """
        Prefactor of the viscosity integral; 3/2 makes sigma_V equal the total cross section for isotropic scattering
        """
        return self._viscosity_normalization

    def _sigma_exact(self, v, log10alpha, log10_mass_ratio):
        """
        sigma_SIDM / m_chi [cm^2 / g] at m_chi = 1 GeV straight from the package, including pair_fraction and
        viscosity_normalization; arguments broadcast
        """
        v, la, lmr = np.broadcast_arrays(np.asarray(v, dtype=float),
                                         np.asarray(log10alpha, dtype=float),
                                         np.asarray(log10_mass_ratio, dtype=float))
        v = np.maximum(v, 1000.0 / 10 ** lmr)          # low velocity plateau
        alpha, v_c = 10 ** la, v / self.C_KMS
        mu = 0.5                                        # reduced mass for m_chi = 1 GeV
        xi, zeta = mu * alpha * 10 ** lmr, alpha / v_c
        se_att, so_att = self._even_odd(xi, zeta, +1)
        se_rep, so_rep = self._even_odd(xi, zeta, -1)
        s = 0.25 * (se_att + so_rep) + 0.75 * (se_rep + so_att)
        sigma_chichibar = 4 * np.pi / (mu * v_c) ** 2 * s * self.GEV2_TO_CM2 / self.GEV_TO_G
        return self._pair_fraction * self._viscosity_normalization * sigma_chichibar

    def _tabulate_velocity(self, v):
        """
        Tabulate log sigma(v) on a (log10 alpha, log10 m_chi/m_phi) grid at fixed v
        :param v: velocity [km / s]
        """
        res = self._cache_resolution
        la_grid = np.arange(self.log10alpha_values[0], self.log10alpha_values[-1] + 0.5 * res, res)
        lmr_grid = np.arange(self.log10_mass_ratio_values[0], self.log10_mass_ratio_values[-1] + 0.5 * res, res)
        LA, LMR = np.meshgrid(la_grid, lmr_grid, indexing='ij')
        log_sigma = np.log(self._sigma_exact(v, LA, LMR))
        self._fixed_velocity_tables[v] = RegularGridInterpolator((la_grid, lmr_grid), log_sigma,
                                                                 bounds_error=False, fill_value=None)

    def _sigma_1gev(self, v, log10alpha, log10_mass_ratio):
        """
        sigma_V / m_chi [cm^2 / g] at m_chi = 1 GeV; from a fixed velocity table if v is a scalar
        """
        if np.ndim(v) == 0 and self._cache_resolution is not None:
            v = float(v)
            if v not in self._fixed_velocity_tables:
                self._tabulate_velocity(v)
            la, lmr = np.broadcast_arrays(np.asarray(log10alpha, dtype=float),
                                          np.asarray(log10_mass_ratio, dtype=float))
            table = self._fixed_velocity_tables[v]
            return np.exp(table((la.ravel(), lmr.ravel()))).reshape(la.shape)
        return self._sigma_exact(v, log10alpha, log10_mass_ratio)

    def __call__(self, v, amp_at_vref, log10alpha, log10_mass_ratio):
        """
        Evaluate the cross section at velocity v
        :param v: velocity [km / s], scalar or array
        :param amp_at_vref: cross section at v_ref [cm^2 / g]; None for the unnormalized value at
                            m_chi = 1 GeV
        :param log10alpha: log10 of the coupling constant alpha_chi
        :param log10_mass_ratio: log10 of m_chi / m_phi
        :return: sigma_V / m_chi [cm^2 / g]
        """
        sigma = self._sigma_1gev(v, log10alpha, log10_mass_ratio)
        if amp_at_vref is not None:
            sigma = sigma * np.asarray(amp_at_vref, dtype=float) / self._sigma_1gev(self._vref, log10alpha,
                                                                                    log10_mass_ratio)
        return sigma.item() if sigma.size == 1 else sigma

    @staticmethod
    def _kernel_weights(v, v0, expon):
        """
        Normalized trapezoid weights on a uniform log10 v grid for the kernel v^(2+expon) exp(-v^2/4v0^2) dv
        :return: shape (len(v), len(v0)); each column sums to one
        """
        w = v[:, None] ** (3 + expon) * np.exp(-0.25 * v[:, None] ** 2 / v0[None, :] ** 2)
        w[0] *= 0.5
        w[-1] *= 0.5
        return w / w.sum(axis=0, keepdims=True)

    def setup_thermal_average_interp(self, log10v0_grid, expon=5, nsteps=101, dlogv=0.01):
        """
        Tabulate <sigma>(v0) / sigma(v_ref) on a regular (log10 v0, log10 alpha, log10 m_chi/m_phi) grid.
        sigma(v) is evaluated once per (alpha, mass ratio) node on one log spaced velocity grid covering
        every v0, and each v0 is a weighted sum over it.
        :param log10v0_grid: 1D array of log10 v0 [km / s]
        :param expon: the power of velocity in the integration kernel
        :param nsteps: grid points along each of the log10 alpha and log10 mass ratio axes
        :param dlogv: velocity sampling [dex]; the package README requires 0.02 or finer
        :return: (points, log10_values, interpolator)
        """
        log10v0_grid = np.asarray(log10v0_grid, dtype=float)
        la_grid = np.linspace(self.log10alpha_values[0], self.log10alpha_values[-1], nsteps)
        lmr_grid = np.linspace(self.log10_mass_ratio_values[0], self.log10_mass_ratio_values[-1], nsteps)
        v = 10 ** np.arange(log10v0_grid.min() - 3.0, log10v0_grid.max() + 1.6 + dlogv, dlogv)
        weights = self._kernel_weights(v, 10 ** log10v0_grid, expon)
        if expon in self._thermal_average_tables:
            print('Overriding the thermal average table for expon=%s... ' % expon)
        values = np.empty((len(log10v0_grid), nsteps, nsteps))
        for i, la in enumerate(la_grid):
            sigma = self._sigma_exact(v[None, :], la, lmr_grid[:, None])        # (nsteps, n_v)
            norm = self._sigma_exact(self._vref, la, lmr_grid)                    # (nsteps,)
            values[:, i, :] = (sigma @ weights).T / norm[None, :]
        log10_values = np.log10(values)
        points = (log10v0_grid, la_grid, lmr_grid)
        self._thermal_average_tables[expon] = RegularGridInterpolator(
            points, log10_values, method='linear', bounds_error=False, fill_value=None)
        return points, log10_values, self._thermal_average_tables[expon]

    def thermal_average(self, v0, amp_at_vref, log10alpha, log10_mass_ratio, expon=5,
                        use_interp=True, dlogv=0.01):
        """
        Thermally averaged cross section for a Maxwell-Boltzmann distribution with dispersion v0
        :param v0: velocity dispersion [km / s]
        :param amp_at_vref: cross section at v_ref [cm^2 / g]
        :param log10alpha: log10 of the coupling constant alpha_chi
        :param log10_mass_ratio: log10 of m_chi / m_phi
        :param expon: the power of velocity in the integration kernel
        :param use_interp: use the table from setup_thermal_average_interp; otherwise integrate directly
        :param dlogv: velocity sampling for direct integration [dex]
        :return: <sigma_V> / m_chi [cm^2 / g]
        """
        v0, amp, la, lmr = [x.ravel() for x in np.broadcast_arrays(
            np.asarray(v0, dtype=float), np.asarray(amp_at_vref, dtype=float),
            np.asarray(log10alpha, dtype=float), np.asarray(log10_mass_ratio, dtype=float))]
        if use_interp:
            if expon not in self._thermal_average_tables:
                raise Exception('no thermal average table for expon=%s; call '
                                'setup_thermal_average_interp(..., expon=%s) first. Built: %s'
                                % (expon, expon, sorted(self._thermal_average_tables)))
            out = 10 ** self._thermal_average_tables[expon](np.column_stack([np.log10(v0), la, lmr])) * amp
        else:
            out = np.empty(v0.size)
            for n in range(v0.size):
                v = 10 ** np.arange(np.log10(v0[n]) - 3.0, np.log10(v0[n]) + 1.6, dlogv)
                w = self._kernel_weights(v, v0[n:n + 1], expon)[:, 0]
                out[n] = amp[n] * np.sum(w * self._sigma_exact(v, la[n], lmr[n])) / \
                         self._sigma_exact(self._vref, la[n], lmr[n])
        return out.item() if out.size == 1 else out

    def compute_mchi(self, log10alpha, log10_mass_ratio, amp_at_vref):
        """
        Dark matter mass for which sigma(v_ref) = amp_at_vref
        :return: m_chi [GeV]
        """
        norm = self._sigma_1gev(self._vref, log10alpha, log10_mass_ratio)
        m_chi = (norm / np.asarray(amp_at_vref, dtype=float)) ** (1.0 / 3.0)
        return m_chi.item() if m_chi.size == 1 else m_chi

    def convert_to_SIDM_params(self, log10alpha, log10_mass_ratio, amp_at_vref):
        """
        :return: m_chi and m_phi [GeV]
        """
        m_chi = self.compute_mchi(log10alpha, log10_mass_ratio, amp_at_vref)
        return m_chi, m_chi / 10 ** np.asarray(log10_mass_ratio)

    def particle_physics_params(self, log10alpha, log10_mass_ratio, amp_at_vref):
        """
        :return: dict with keys 'm_chi' (GeV), 'm_phi' (GeV), 'alpha_chi'
        """
        m_chi, m_phi = self.convert_to_SIDM_params(log10alpha, log10_mass_ratio, amp_at_vref)
        return {'m_chi': m_chi, 'm_phi': m_phi, 'alpha_chi': 10 ** np.asarray(log10alpha)}

    def sigma_from_particle_params(self, v, m_chi, m_phi, alpha_chi):
        """
        Viscosity cross section per unit mass, including pair_fraction and viscosity_normalization, for
        physical particle parameters, evaluated directly from the yukawa_viscosity_even_odd tables. Unlike
        __call__, this never uses the fixed velocity or thermal average lookup tables, so it is not
        restricted to log10alpha_values and log10_mass_ratio_values: it is defined wherever the package has
        tabulated values or its CLASSICS fallback applies, and is nan elsewhere. All arguments broadcast
        against each other; the package is memory hungry, so this is meant for plotting grids rather than
        very large sample arrays.
        :param v: relative velocity [km / s]
        :param m_chi: dark matter mass [GeV]
        :param m_phi: mediator mass [GeV]
        :param alpha_chi: the coupling constant
        :return: sigma_V / m_chi [cm^2 / g]
        """
        m_chi = np.asarray(m_chi, dtype=float)
        log10_mass_ratio = np.log10(m_chi / np.asarray(m_phi, dtype=float))
        # the package values are for m_chi = 1 GeV, and sigma/m_chi scales as m_chi^-3
        sigma = self._sigma_exact(v, np.log10(alpha_chi), log10_mass_ratio) * m_chi ** -3
        return sigma.item() if sigma.size == 1 else sigma

    def fixed_alpha_thermal_average(self, log10_alpha, log10v0_grid, log10_mass_ratio_range,
                                    dlog10_mass_ratio=0.001, expon=5, dlogv=0.01):
        """
        Build an interpolator for the thermally averaged cross section at fixed alpha_chi, evaluated
        directly from the yukawa_viscosity_even_odd tables, so the mass ratio range is not restricted to
        log10_mass_ratio_values. At fixed alpha_chi and m_chi / m_phi, <sigma_V> / m_chi scales as
        m_chi^-3, so the table is two dimensional, in (log10 v0, log10 m_chi/m_phi) at m_chi = 1 GeV.
        sigma(v) is computed once per mass ratio on one log spaced velocity grid covering every v0, and
        each v0 is a weighted sum over it.

        Resonances sit at fixed xi = alpha_chi (m_chi / m_phi) / 2, so at fixed alpha_chi they are narrow
        features in the mass ratio that the thermal average does not smooth; dlog10_mass_ratio sets how
        well they are resolved. A mass ratio is nan in the table only if the package has no data at a
        velocity where the kernel carries weight.
        :param log10_alpha: log10 of the coupling constant, held fixed
        :param log10v0_grid: 1D array of at least two log10 v0 values [km / s]
        :param log10_mass_ratio_range: (min, max) of log10 m_chi / m_phi to tabulate
        :param dlog10_mass_ratio: spacing of the mass ratio grid [dex]
        :param expon: the power of velocity in the integration kernel
        :param dlogv: velocity sampling [dex]; the package README requires 0.02 or finer
        :return: a FixedAlphaThermalAverage, called as f(v0, m_chi, m_phi) with masses in GeV
        """
        log10v0_grid = np.asarray(log10v0_grid, dtype=float)
        lmr_grid = np.arange(log10_mass_ratio_range[0], log10_mass_ratio_range[1] + 0.5 * dlog10_mass_ratio,
                             dlog10_mass_ratio)
        v = 10 ** np.arange(log10v0_grid.min() - 3.0, log10v0_grid.max() + 1.6 + dlogv, dlogv)
        weights = self._kernel_weights(v, 10 ** log10v0_grid, expon)
        # velocities where the kernel carries weight. A value missing where the weight is below 1e-8 of its peak
        # changes the average by <~ 1e-4 even if sigma grows toward low velocity as steeply as v^-4 (the Born
        # limit); this happens at the top of the package's lattice (xi > 1e5), where CLASSICS needs kappa > 10
        support = (weights > 1e-8 * weights.max(axis=0, keepdims=True)).astype(float)
        values = np.empty((len(log10v0_grid), len(lmr_grid)))
        chunk = 200                                      # bounds the memory of each package call
        for start in range(0, len(lmr_grid), chunk):
            sigma = self._sigma_exact(v[None, :], log10_alpha, lmr_grid[start:start + chunk, None])
            finite = np.isfinite(sigma)
            block = (np.where(finite, sigma, 0.0) @ weights).T
            block[((~finite).astype(float) @ support).T > 0] = np.nan
            values[:, start:start + chunk] = block
        return FixedAlphaThermalAverage(self, log10_alpha, expon, log10v0_grid, lmr_grid, np.log10(values))

    def relic_density_log10_dsphr(self, log10alpha, log10_mass_ratio, relic_coeff=0.035):
        """
        log10 sigma(v_ref) [cm^2 / g] for which alpha_chi = relic_coeff * (m_chi / TeV)
        """
        log10alpha = np.asarray(log10alpha, dtype=float)
        log10_mchi = log10alpha + 3.0 - np.log10(relic_coeff)
        out = np.log10(self._sigma_1gev(self._vref, log10alpha, log10_mass_ratio)) - 3.0 * log10_mchi
        return out.item() if out.size == 1 else out

class FixedAlphaThermalAverage(object):
    """
    Thermally averaged viscosity cross section at fixed alpha_chi as a function of (v0, m_chi, m_phi),
    built by FermionChargedScalarCrossSection.fixed_alpha_thermal_average. The table is in
    (log10 v0, log10 m_chi/m_phi) at m_chi = 1 GeV, and each call rescales by m_chi^-3. The cross section
    at a single velocity, at the same alpha_chi, is available from sigma().
    """

    def __init__(self, cross_section, log10_alpha, expon, log10v0_grid, log10_mass_ratio_grid, log10_values):
        """
        :param cross_section: the FermionChargedScalarCrossSection the table was built from
        :param log10_alpha: log10 of the coupling constant the table was built at
        :param expon: the power of velocity in the integration kernel
        :param log10v0_grid: log10 v0 [km / s], shape (n_v0,)
        :param log10_mass_ratio_grid: log10 m_chi / m_phi, shape (n_mr,)
        :param log10_values: log10 <sigma_V> / m_chi [cm^2 / g] at m_chi = 1 GeV, shape (n_v0, n_mr)
        """
        self._cross_section = cross_section
        self.log10_alpha = log10_alpha
        self.expon = expon
        self.log10v0_grid = log10v0_grid
        self.log10_mass_ratio_grid = log10_mass_ratio_grid
        self.log10_values = log10_values
        self._interp = RegularGridInterpolator((log10v0_grid, log10_mass_ratio_grid), log10_values,
                                               bounds_error=False, fill_value=np.nan)

    def __call__(self, v0, m_chi, m_phi):
        """
        Evaluate the thermally averaged cross section; all arguments broadcast against each other
        :param v0: velocity dispersion [km / s]
        :param m_chi: dark matter mass [GeV]
        :param m_phi: mediator mass [GeV]
        :return: <sigma_V> / m_chi [cm^2 / g]; nan outside the tabulated v0 and mass ratio ranges
        """
        v0, m_chi, m_phi = np.broadcast_arrays(np.asarray(v0, dtype=float), np.asarray(m_chi, dtype=float),
                                               np.asarray(m_phi, dtype=float))
        log10_sigma_1gev = self._interp((np.log10(v0).ravel(), np.log10(m_chi / m_phi).ravel()))
        out = 10 ** log10_sigma_1gev.reshape(v0.shape) * m_chi ** -3
        return out.item() if out.size == 1 else out

    def sigma(self, v, m_chi, m_phi):
        """
        Cross section at a single relative velocity, at this alpha_chi, evaluated directly from the
        package (see FermionChargedScalarCrossSection.sigma_from_particle_params); for constraints such
        as the cluster and evaporation limits. All arguments broadcast against each other.
        :param v: relative velocity [km / s]
        :param m_chi: dark matter mass [GeV]
        :param m_phi: mediator mass [GeV]
        :return: sigma_V / m_chi [cm^2 / g]
        """
        return self._cross_section.sigma_from_particle_params(v, m_chi, m_phi, 10 ** self.log10_alpha)


class CrossSectionInterpolator(object):
    """
    Interpolates over a grid of CrossSection instances to give a continuous
    representation of σ(v; log10α, log10(m_χ/m_φ)). Useful for parameter
    inference where the cross section must be evaluated at many parameter combinations.
    """

    def __init__(self, v, log10alpha_values, log10_mass_ratio_values, cross_section_table,
                 interp_method='linear', v_ref=100):
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
        self._thermal_average_tables = {}

    @property
    def v_ref(self):
        """
        Reference velocity in km/s
        """
        return self._vref

    @classmethod
    def from_file(cls, base_path, v, log10alpha_values, log10_mass_ratio_values,
                  cross_section_type='VISCOSITY', v_power=4.0, num_decimal=2, v_ref=100):
        """
        Build a CrossSectionInterpolator by loading a grid of phase shift files and
        evaluating each cross section on the velocity array v.

        :param base_path:                path to directory containing phase shift files
        :param v:                        velocities in km/s at which to evaluate sigma, shape (n_v,)
        :param log10alpha_values:        grid of log10(alpha) values, shape (n_alpha,)
        :param log10_mass_ratio_values:  grid of log10(m_chi/m_phi) values, shape (n_mr,)
        :param cross_section_type:       'VISCOSITY', 'MOMENTUM', or 'TOTAL'
        :param v_power:                  power law index for high-velocity extrapolation
        :param num_decimal:              decimal places encoded in the filename (2 or 3)
        :returns:                        CrossSectionInterpolator instance
        """
        cross_section_table = np.empty((len(log10alpha_values),
                                        len(log10_mass_ratio_values),
                                        len(v)))
        for i, log10alpha in enumerate(log10alpha_values):
            for j, log10_mass_ratio in enumerate(log10_mass_ratio_values):
                cross = CrossSection.from_file(base_path,
                                               log10alpha,
                                               log10_mass_ratio,
                                               cross_section_type=cross_section_type,
                                               v_power=v_power,
                                               num_decimal=num_decimal)
                cross_section_table[i, j, :] = cross(v, None)
        return cls(v, log10alpha_values, log10_mass_ratio_values, cross_section_table,
                   v_ref=v_ref)

    def setup_thermal_average_interp(self, log10v0_grid, expon=5, nsteps=101):
        """
        Precompute a 3D interpolation table for the thermal average ⟨σ⟩(v0; log10α, log10(m_χ/m_φ))
        on a regular grid, using parallel evaluation. After calling this method,
        thermal_average() will use the interpolator instead of direct integration.

        :param log10v0_grid: 1D array of log10(v0) values defining the v0 grid
        :param nsteps:       number of grid points along each of the log10α and
                             log10(m_χ/m_φ) axes (default 20)
        :returns:            tuple (points, log10_values, interpolator)
        """

        amp_at_vref_ref = 1.0
        nsteps_v0 = len(log10v0_grid)
        log10alpha_grid = np.linspace(self.log10alpha_values[0],
                                      self.log10alpha_values[-1],
                                      nsteps)
        log10_mass_ratio_grid = np.linspace(self.log10_mass_ratio_values[0],
                                            self.log10_mass_ratio_values[-1],
                                            nsteps)
        if expon in self._thermal_average_tables:
            print('Overriding the thermal average table for expon=%s... ' % expon)

        indices = [
            (i, j, k, lv0, lalpha, lmr)
            for i, lv0 in enumerate(log10v0_grid)
            for j, lalpha in enumerate(log10alpha_grid)
            for k, lmr in enumerate(log10_mass_ratio_grid)
        ]

        def _eval(i, j, k, lv0, lalpha, lmr):
            val = self.thermal_average(v0=10 ** lv0, amp_at_vref=amp_at_vref_ref,
                                       log10alpha=lalpha, log10_mass_ratio=lmr,
                                       expon=expon,
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
        self._thermal_average_tables[expon] = RegularGridInterpolator(
            points, log10_values, method='linear', bounds_error=False, fill_value=None)

        return points, log10_values, self._thermal_average_tables[expon]

    def _thermal_average_interpolation(self, v0, amp_at_vref, log10alpha, log10_mass_ratio, expon):
        """
        Evaluate the precomputed thermal average interpolation table.
        Raises an exception if setup_thermal_average_interp() has not been called.

        :param v0:               velocity dispersion in km/s
        :param amp_at_vref:      cross section amplitude in cm²/g at v_ref
        :param log10alpha:       log10 of the coupling constant α
        :param log10_mass_ratio: log10 of the mass ratio m_χ / m_φ
        :param expon:       the power of velocity in the integration kernel
        :returns:                thermally averaged cross section in cm²/g
        """
        if expon not in self._thermal_average_tables:
            raise Exception('no thermal average table for expon=%s; call '
                            'setup_thermal_average_interp(..., expon=%s) first. Built: %s'
                            % (expon, expon, sorted(self._thermal_average_tables)))

        v0 = np.atleast_1d(v0)
        log10alpha = np.atleast_1d(log10alpha)
        log10_mass_ratio = np.atleast_1d(log10_mass_ratio)
        amp_at_vref = np.atleast_1d(amp_at_vref)
        point = np.column_stack([np.log10(v0), log10alpha, log10_mass_ratio])
        log10_result = self._thermal_average_tables[expon](point)
        result = 10 ** log10_result * amp_at_vref
        return float(result[0]) if result.size == 1 else result

    def thermal_average(self, v0, amp_at_vref, log10alpha, log10_mass_ratio, expon=5,
                        use_interp=True, num_steps=200):
        """
        Compute the thermally averaged cross section ⟨σ⟩ for a Maxwell-Boltzmann
        distribution with velocity dispersion v0. Can use either direct numerical
        integration or the precomputed interpolation table.

        :param v0:               velocity dispersion in km/s
        :param amp_at_vref:      cross section amplitude in cm²/g at v_ref
        :param log10alpha:       log10 of the coupling constant α
        :param log10_mass_ratio: log10 of the mass ratio m_χ / m_φ
        :param expon:       the power of velocity in the integration kernel
        :param use_interp:       if True, use the precomputed interpolation table
                                 (requires setup_thermal_average_interp() to have been called);
                                 if False, perform direct numerical integration (slower)
        :param num_steps:        number of velocity integration points when use_interp=False
        :returns:                thermally averaged cross section in cm²/g
        """
        if use_interp:
            return self._thermal_average_interpolation(v0, amp_at_vref, log10alpha, log10_mass_ratio, expon)
        else:
            _v = np.logspace(np.log10(0.05 * v0), np.log10(150 * v0), num_steps)
            _sigmav = self._evaluate_cross_section(_v, amp_at_vref, log10alpha, log10_mass_ratio)
            kernel = _v ** (2+expon) * np.exp(-0.25 * _v ** 2 / v0 ** 2)
            return np.trapezoid(kernel * _sigmav, _v) / np.trapezoid(kernel, _v)

    def thermal_average_integrand(self, v0, amp_at_vref, log10alpha, log10_mass_ratio, expon=5):
        """
        Return the integrand of the thermal average for diagnostic purposes.

        :param v0:               velocity dispersion in km/s
        :param amp_at_vref:      cross section amplitude in cm²/g at v_ref
        :param log10alpha:       log10 of the coupling constant α
        :param log10_mass_ratio: log10 of the mass ratio m_χ / m_φ
        :param expon:            the power of velocity in the integration kernel
        :returns:                tuple (_v, integrand) where _v is the velocity array in km/s
                                 and integrand = σ(v) * v^7 * exp(-v²/4v0²)
        """
        _v = np.logspace(np.log10(0.01 * v0), np.log10(100 * v0), 2000)
        _sigmav = self._evaluate_cross_section(_v, amp_at_vref, log10alpha, log10_mass_ratio)
        kernel = _v ** (2+expon) * np.exp(-0.25 * _v ** 2 / v0 ** 2)
        return _v, kernel * _sigmav

    def relic_density_log10_dsphr(self, log10alpha, log10_mass_ratio, relic_coeff=0.035):
        """
        Compute the value of log10(sigma/m_chi) at v_ref [cm^2/g] that satisfies the
        thermal relic density constraint alpha_chi ~ 0.1 * (m_chi / TeV), given
        log10alpha and log10_mass_ratio.

        :param log10alpha:        log10 of coupling constant alpha_chi, scalar or array
        :param log10_mass_ratio:  log10 of m_chi / m_phi, scalar or array
        :returns:                 log10(amp_at_vref) satisfying the relic constraint
        """
        log10alpha, log10_mass_ratio = np.broadcast_arrays(np.atleast_1d(log10alpha),
                                                           np.atleast_1d(log10_mass_ratio))
        points = np.column_stack([log10alpha.ravel(), log10_mass_ratio.ravel(),
                                  np.full(log10alpha.size, np.log(self._vref))])
        log10_m_chi = log10alpha.ravel() - np.log10(relic_coeff) + 3.0
        log10_amp = self.interp(points) / np.log(10.0) - 3.0 * log10_m_chi
        return float(log10_amp[0]) if log10_amp.size == 1 else log10_amp

    def convert_to_SIDM_params(self, log10alpha, log10_mass_ratio, amp_at_vref):
        """
        Compute the m_chi, m_phi values corresponding to a set of alpha_chi, m_chi / m_phi, amp_at_vref values
        return: m_chi and m_phi in Gev
        """
        m_chi = self.compute_mchi(log10alpha, log10_mass_ratio, amp_at_vref)
        m_phi = m_chi / 10 ** log10_mass_ratio
        return m_chi, m_phi

    def compute_mchi(self, log10alpha, log10_mass_ratio, amp_at_vref):
        """
        Compute the DM particle mass m_chi for a given alpha, m_chi / m_phi, and amp_at_vref combination
        return m_chi in GeV
        """
        xnorm = (log10alpha, log10_mass_ratio, np.log(self._vref))
        m_chi = np.exp((self.interp(xnorm) - np.log(amp_at_vref)) / 3)
        return m_chi

    def particle_physics_params(self, log10alpha, log10_mass_ratio, amp_at_vref):
        """
        Infer the particle physics parameters (m_χ, m_φ, α_χ) that produce a cross section
        of amp_at_vref cm²/g at velocity v_ref km/s, for given log10α and log10(m_χ/m_φ).

        :param log10alpha:       log10 of the coupling constant α
        :param log10_mass_ratio: log10 of the mass ratio m_χ / m_φ
        :param amp_at_vref:      desired cross section in cm²/g at v_ref
        :returns:                dict with keys 'm_chi' (GeV), 'm_phi' (GeV), 'alpha_chi'
        """
        m_chi, m_phi = self.convert_to_SIDM_params(log10alpha, log10_mass_ratio,
                                                   amp_at_vref)
        alpha_chi = 10 ** log10alpha
        params = {'m_chi': m_chi,
                  'm_phi': m_phi,
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
        if amp_at_vref is None:
            return np.exp(self.interp(x))
        xnorm = (log10alpha, log10_mass_ratio, np.log(self._vref))
        return amp_at_vref * np.exp(self.interp(x) - self.interp(xnorm))

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

class MixtureCrossSectionInterpolator(object):
    """
    A weighted mixture of two CrossSectionInterpolator instances, e.g. a 50/50 population of
    attractive and repulsive Yukawa scatterers sharing a single m_chi.

    The components are combined BEFORE normalization: the unnormalized cross sections (both
    tabulated at m_chi = 1 GeV) are averaged, and the mixture is then scaled so that
    sigma(v_ref) = amp_at_vref. This is the correct convention for a single dark matter mass --
    normalizing each component separately would implicitly assign each a different m_chi.
    """

    def __init__(self, components, weights=None):
        """
        :param components: list of CrossSectionInterpolator instances
        :param weights: relative weights; defaults to equal weighting. Normalized internally.
        """
        if weights is None:
            weights = np.ones(len(components))
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(components):
            raise Exception('got %d components but %d weights' % (len(components), len(weights)))
        if np.any(weights < 0):
            raise Exception('weights must be non-negative')
        self.components = list(components)
        self.weights = weights

        v_refs = [c.v_ref for c in self.components]
        if not np.allclose(v_refs, v_refs[0]):
            raise Exception('components must share a reference velocity, got ' + str(v_refs))
        self._vref = v_refs[0]

        # the mixture is only defined where every component is tabulated
        self.log10alpha_values = np.array(
            [max(min(c.log10alpha_values) for c in self.components),
             min(max(c.log10alpha_values) for c in self.components)])
        self.log10_mass_ratio_values = np.array(
            [max(min(c.log10_mass_ratio_values) for c in self.components),
             min(max(c.log10_mass_ratio_values) for c in self.components)])

    @property
    def v_ref(self):
        """
        Reference velocity in km/s
        """
        return self._vref

    def _unnormalized_at_vref(self, log10alpha, log10_mass_ratio):
        """
        Unnormalized cross section of each component at v_ref, at m_chi = 1 GeV
        """
        return [np.atleast_1d(c(self._vref, None, log10alpha, log10_mass_ratio))
                for c in self.components]

    def _mixture_norm(self, log10alpha, log10_mass_ratio):
        """
        Weighted mean of the components' unnormalized cross sections at v_ref. This is what
        amp_at_vref is matched against, and what sets m_chi.
        """
        at_vref = self._unnormalized_at_vref(log10alpha, log10_mass_ratio)
        return sum(w * s for w, s in zip(self.weights, at_vref))

    def relic_density_log10_dsphr(self, log10alpha, log10_mass_ratio, relic_coeff=0.035):
        """
        log10(sigma/m_chi) at v_ref [cm^2/g] for which the relic condition
        alpha_chi = relic_coeff * (m_chi / TeV) holds, at fixed (log10alpha, log10_mass_ratio).
        Uses the mixture normalization at m_chi = 1 GeV, so it is consistent with compute_mchi.
        """
        log10alpha, log10_mass_ratio = np.broadcast_arrays(np.atleast_1d(log10alpha),
                                                           np.atleast_1d(log10_mass_ratio))
        log10_amp_1 = np.log10(self._mixture_norm(log10alpha, log10_mass_ratio))   # m_chi = 1 GeV
        log10_mchi = log10alpha + 3.0 - np.log10(relic_coeff)    # m_chi [GeV] = 1e3 * alpha / relic_coeff
        out = log10_amp_1 - 3.0 * log10_mchi
        return float(out[0]) if out.size == 1 else out

    def __call__(self, v, amp_at_vref, log10alpha, log10_mass_ratio):
        """
        Evaluate the mixture cross section
        :param v: velocity in km/s, scalar or array
        :param amp_at_vref: desired mixture cross section in cm^2/g at v_ref; None for unnormalized
        :param log10alpha: log10 of the coupling constant alpha
        :param log10_mass_ratio: log10 of the mass ratio m_chi / m_phi
        :return: cross section in cm^2/g
        """
        total = sum(w * np.atleast_1d(c(v, None, log10alpha, log10_mass_ratio))
                    for w, c in zip(self.weights, self.components))
        if amp_at_vref is None:
            out = total
        else:
            out = total * np.atleast_1d(amp_at_vref) / self._mixture_norm(log10alpha, log10_mass_ratio)
        return float(out[0]) if out.size == 1 else out

    def thermal_average(self, v0, amp_at_vref, log10alpha, log10_mass_ratio, expon=5,
                        use_interp=True, num_steps=200):
        """
        Thermally averaged mixture cross section. Thermal averaging is linear in sigma, so this is
        the weighted mean of the components' unnormalized thermal averages, rescaled by the same
        factor that normalizes the mixture at v_ref.
        :param v0: velocity dispersion in km/s
        :param amp_at_vref: desired mixture cross section in cm^2/g at v_ref
        :param log10alpha: log10 of the coupling constant alpha
        :param log10_mass_ratio: log10 of the mass ratio m_chi / m_phi
        :param expon:       the power of velocity in the integration kernel
        :param use_interp: use each component's precomputed thermal average table
        :param num_steps: velocity integration points when use_interp is False
        :return: thermally averaged cross section in cm^2/g
        """
        at_vref = self._unnormalized_at_vref(log10alpha, log10_mass_ratio)
        total = 0.0
        for w, c, s_vref in zip(self.weights, self.components, at_vref):
            # calling with amp=1 returns <sigma>/sigma(v_ref), so multiply back to unnormalize
            shape = np.atleast_1d(c.thermal_average(v0,
                                                    1.0,
                                                    log10alpha,
                                                    log10_mass_ratio,
                                                    expon=expon,
                                                    use_interp=use_interp,
                                                    num_steps=num_steps))
            total = total + w * shape * s_vref
        out = total * np.atleast_1d(amp_at_vref) / self._mixture_norm(log10alpha, log10_mass_ratio)
        return float(out[0]) if out.size == 1 else out

    def setup_thermal_average_interp(self, log10v0_grid, expon=5, nsteps=101):
        """
        Build the thermal average interpolation table for every component
        :param log10v0_grid: 1D array of log10(v0) values
        :param expon: the power of velocity in the integration kernel
        :param nsteps: grid points along each of the log10alpha and log10 mass ratio axes
        :return: list of each component's (points, log10_values, interpolator)
        """
        return [c.setup_thermal_average_interp(log10v0_grid, expon=expon, nsteps=nsteps)
                for c in self.components]

    def compute_mchi(self, log10alpha, log10_mass_ratio, amp_at_vref):
        """
        Dark matter mass that gives the mixture a cross section of amp_at_vref at v_ref. The
        tabulated cross sections assume m_chi = 1 GeV and scale as m_chi^-3.
        :return: m_chi in GeV
        """
        norm = self._mixture_norm(log10alpha, log10_mass_ratio)
        m_chi = (norm / np.atleast_1d(amp_at_vref)) ** (1.0 / 3.0)
        return float(m_chi[0]) if m_chi.size == 1 else m_chi

    def particle_physics_params(self, log10alpha, log10_mass_ratio, amp_at_vref):
        """
        Infer (m_chi, m_phi, alpha_chi) for the mixture
        :return: dict with keys 'm_chi' (GeV), 'm_phi' (GeV), 'alpha_chi'
        """
        m_chi = self.compute_mchi(log10alpha, log10_mass_ratio, amp_at_vref)
        return {'m_chi': m_chi,
                'm_phi': m_chi / 10 ** log10_mass_ratio,
                'alpha_chi': 10 ** log10alpha}

_TABLES = {}
def _load_table(potential):
    if potential not in _TABLES:
        with np.load(files("pykawa.data").joinpath(f"{potential}_phases.npz")) as d:
            _TABLES[potential] = {k: d[k] for k in d.files}
    return _TABLES[potential]

from pykawa.semiclassical import sigmaV_high_velocity

_RSHROD_UNIT_RATIO = 0.998929   # sigma(exact constants) / sigma(pykawa's unit constants), to join the tail to the table


def _load_rshrod_table(potential):
    """lattice tables: pykawa/data/<potential>.npz (no '_phases' suffix), cached in _TABLES"""
    if potential not in _TABLES:
        with np.load(files("pykawa.data").joinpath(f"{potential}.npz")) as d:
            _TABLES[potential] = {k: d[k] for k in d.files}
    return _TABLES[potential]


def make_cross_section_interpolator(potential, cross_section_type='VISCOSITY',
                                    v=None, v_power=4.0, v_ref=100, weights=None):
    """
    Build a CrossSectionInterpolator from the precomputed data for a given potential.

    ATTRACTIVE_YUKAWA_RSHROD is the viscosity cross section for scattering of a Dirac fermion
    with its antiparticle (chi chi-bar), which are distinguishable: all partial waves contribute,
    sigma_V = S(all l). It is the correct attractive channel for a vector mediator, where the
    potential is attractive between chi and chi-bar and repulsive between two chi.

    It does NOT apply to attractive scattering of identical particles, for example chi chi
    scattering through a scalar mediator (attractive for every pair). Identical fermions require
    the spin-averaged combination sigma_V = sigma_V^even/4 + 3 sigma_V^odd/4, computed from the
    even- and odd-l partial waves of the attractive potential separately. Those tables are not
    included. The two cases differ most where few partial waves contribute (low velocity,
    near resonances), by up to a factor of 4 in the s-wave regime.

    REPULSIVE_YUKAWA_RSHROD is chi chi scattering of identical Dirac fermions, already spin
    averaged as above. MIXTURE_RSHROD is the equal-weight average of the two, i.e. the
    cross section of a symmetric chi / chi-bar population.

    FERMION_CHARGED_SCALAR is chi chi-bar scattering of fermionic dark matter with a charged scalar
    mediator, assembled from the parity-resolved partial wave sums of yukawa_viscosity_even_odd; see
    FermionChargedScalarCrossSection. It is evaluated directly rather than from a lattice of models,
    so v, v_power and weights are ignored. It returns the pair-averaged cross section of a symmetric
    population with the 3/2 viscosity normalization, sigma_SIDM = (1/2)(3/2) sigma_chi-chibar; for other
    choices, construct FermionChargedScalarCrossSection directly with pair_fraction and
    viscosity_normalization.

    Parameters
    ----------
    potential : str
        'ATTRACTIVE_YUKAWA', 'REPULSIVE_YUKAWA' or 'MIXTURE' (phase-shift tables), or
        'ATTRACTIVE_YUKAWA_RSHROD', 'REPULSIVE_YUKAWA_RSHROD' or 'MIXTURE_RSHROD' (lattice of exact
        sigma_V(v) curves: 87 x 87 models, log10 alpha = -5.20..-1.76 step 0.04,
        log10 m_chi/m_phi = 2.80103..5.38103 step 0.03; VISCOSITY only), or
        'FERMION_CHARGED_SCALAR' (VISCOSITY only).
    cross_section_type : str, optional
        One of 'VISCOSITY', 'MOMENTUM', or 'TOTAL'. Default is 'VISCOSITY'.
    v : ndarray, optional
        Velocity array in km/s at which to evaluate the cross section.
        Defaults to np.logspace(-1, 4, 100); for the RSHROD tables to np.logspace(-2, 4, 601)
        (0.01 dex, the resolution of the exact curves).
    v_power : float, optional
        Power law index for high-velocity extrapolation. Default is 4.0.
    v_ref: reference velocity at which to normalize cross section strength to "amp_at_vref"

    Returns
    -------
    CrossSectionInterpolator
    """
    if potential == 'FERMION_CHARGED_SCALAR':
        if cross_section_type != 'VISCOSITY':
            raise ValueError(f'{potential} is sigma_V only; cross_section_type must be VISCOSITY')
        cross_section_interpolator = FermionChargedScalarCrossSection(v_ref=v_ref)
        return (cross_section_interpolator,
                None,
                cross_section_interpolator.log10_mass_ratio_values,
                cross_section_interpolator.log10alpha_values)

    rshrod = potential.endswith('_RSHROD')
    if v is None:
        v = np.logspace(-2, 4, 601) if rshrod else np.logspace(-1, 4, 100)

    if potential in ('MIXTURE', 'MIXTURE_RSHROD'):
        suffix = '_RSHROD' if rshrod else ''
        cross_section_interpolator_att, _, _, _ = make_cross_section_interpolator('ATTRACTIVE_YUKAWA' + suffix,
                                                                                  cross_section_type,
                                                                                  v, v_power, v_ref)
        cross_section_interpolator_rep, _, _, _ = make_cross_section_interpolator('REPULSIVE_YUKAWA' + suffix,
                                                                                  cross_section_type,
                                                                                  v, v_power, v_ref)
        return (MixtureCrossSectionInterpolator([cross_section_interpolator_att,
                                                 cross_section_interpolator_rep],
                                                weights=weights),
                None,
                cross_section_interpolator_att.log10_mass_ratio_values,
                cross_section_interpolator_att.log10alpha_values)

    if rshrod:
        if cross_section_type != 'VISCOSITY':
            raise ValueError(f'{potential} stores sigma_V only; cross_section_type must be VISCOSITY')
        data = _load_rshrod_table(potential)
        attractive = str(data['potential']) == 'attractive'
        particles = str(data.get('particles', 'distinguishable'))
        log10alpha_values = data['log10_alpha']
        log10_mass_ratio_values = data['log10_mass_ratio']
        cross_section_table = np.empty((len(log10alpha_values), len(log10_mass_ratio_values), len(v)))
        for i, log10alpha in enumerate(log10alpha_values):
            for j, log10_mass_ratio in enumerate(log10_mass_ratio_values):
                n = int(data['n_v'][i, j])                                   # lattice tables are indexed [alpha, mass ratio]
                v_tab = 10.0 ** data['log10_v'][i, j, :n]
                sigma_tab = 10.0 ** data['log10_sigma_V'][i, j, :n]
                if v_tab[-1] < 8.5e3:                                        # exact up to v_edge; continue with Born / semi-classical
                    v_add = np.logspace(np.log10(v_tab[-1]) + 0.01, np.log10(9.0e3), 40)
                    s_add = _RSHROD_UNIT_RATIO * sigmaV_high_velocity(v_add,
                                                                      log10alpha,
                                                                      log10_mass_ratio,
                                                                      attractive=attractive,
                                                                      particles=particles)
                    v_tab, sigma_tab = np.append(v_tab, v_add), np.append(sigma_tab, s_add)
                cross = CrossSection(v_tab, sigma_tab, v_power, log10alpha, log10_mass_ratio)
                cross_section_table[i, j, :] = cross(v, None)
        cross_section_interpolator = CrossSectionInterpolator(v, log10alpha_values, log10_mass_ratio_values,
                                                              cross_section_table, v_ref=v_ref)
        return cross_section_interpolator, cross_section_table, log10_mass_ratio_values, log10alpha_values

    scale = 10 if potential == 'REPULSIVE_YUKAWA' else 100
    data = _load_table(potential)
    mphi_vals = data["mphi"]
    alpha_vals = data["alpha"]
    log10_mass_ratio_values = mphi_vals / scale
    log10alpha_values = -alpha_vals / scale
    n_alpha = len(log10alpha_values)
    n_mr = len(log10_mass_ratio_values)
    cross_section_table = np.empty((n_alpha, n_mr, len(v)))
    for i, log10alpha in enumerate(log10alpha_values):
        for j, log10_mass_ratio in enumerate(log10_mass_ratio_values):
            v_grid = data["v"][j, i].astype(np.float64)        # tables are indexed [mass ratio, alpha]
            phases = data["phases"][j, i].astype(np.float64)
            cross = CrossSection.from_phase_shifts(v_grid, phases, cross_section_type,
                                                   log10alpha, log10_mass_ratio, v_power=v_power)
            cross_section_table[i, j, :] = cross(v, None)
    cross_section_interpolator = CrossSectionInterpolator(v, log10alpha_values, log10_mass_ratio_values,
                                                          cross_section_table,
                                                          v_ref=v_ref)
    return cross_section_interpolator, cross_section_table, log10_mass_ratio_values, log10alpha_values
