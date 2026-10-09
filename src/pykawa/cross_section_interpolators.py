from scipy.interpolate import RegularGridInterpolator
import numpy as np
from pykawa.cross_sections import CrossSection
from joblib import Parallel, delayed
from importlib.resources import files
from tqdm import tqdm

class CrossSectionInterpolator(object):
    """
    Interpolates over a grid of CrossSection instances to give a continuous
    representation of σ(v; log10α, log10(m_χ/m_φ)). Useful for parameter
    inference where the cross section must be evaluated at many parameter combinations.
    """

    def __init__(self, v, log10alpha_values, log10_mass_ratio_values, cross_section_table,
                 interp_method='linear', v_ref=60, thermal_average_interpolation_class=None):
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

    @property
    def v_ref(self):
        """
        Reference velocity in km/s
        """
        return self._vref

    @classmethod
    def from_file(cls, base_path, v, log10alpha_values, log10_mass_ratio_values,
                  cross_section_type='VISCOSITY', v_power=4.0, num_decimal=2, v_ref=60):
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

    def setup_thermal_average_interp(self, log10v0_grid, expon=7, nsteps=101):
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
        log10alpha_grid = np.linspace(self.log10alpha_values[0],
                                      self.log10alpha_values[-1],
                                      nsteps)
        log10_mass_ratio_grid = np.linspace(self.log10_mass_ratio_values[0],
                                            self.log10_mass_ratio_values[-1],
                                            nsteps)

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
        return float(result[0]) if result.size == 1 else result

    def thermal_average(self, v0, amp_at_vref, log10alpha, log10_mass_ratio, expon=7,
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
            kernel = _v ** expon * np.exp(-0.25 * _v ** 2 / v0 ** 2)
            return np.trapezoid(kernel * _sigmav, _v) / np.trapezoid(kernel, _v)

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
        self.weights = weights / weights.sum()

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

    def thermal_average(self, v0, amp_at_vref, log10alpha, log10_mass_ratio, expon=7,
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
            shape = np.atleast_1d(c.thermal_average(v0, 1.0, log10alpha, log10_mass_ratio,
                                                    expon=expon, use_interp=use_interp, num_steps=num_steps))
            total = total + w * shape * s_vref
        out = total * np.atleast_1d(amp_at_vref) / self._mixture_norm(log10alpha, log10_mass_ratio)
        return float(out[0]) if out.size == 1 else out

    def setup_thermal_average_interp(self, log10v0_grid, nsteps=61):
        """
        Build the thermal average interpolation table for every component
        :param log10v0_grid: 1D array of log10(v0) values
        :param nsteps: grid points along each of the log10alpha and log10 mass ratio axes
        :return: list of each component's (points, log10_values, interpolator)
        """
        return [c.setup_thermal_average_interp(log10v0_grid, nsteps=nsteps)
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

def make_cross_section_interpolator(potential, cross_section_type='VISCOSITY',
                                    v=None, v_power=4.0, v_ref=60):
    """
    Build a CrossSectionInterpolator from the precomputed phase shift data
    for a given potential.

    Parameters
    ----------
    potential : str
        Either 'ATTRACTIVE_YUKAWA' or 'REPULSIVE_YUKAWA'.
    cross_section_type : str, optional
        One of 'VISCOSITY', 'MOMENTUM', or 'TOTAL'. Default is 'VISCOSITY'.
    v : ndarray, optional
        Velocity array in km/s at which to evaluate the cross section.
        Defaults to np.logspace(-1, 4, 100).
    v_power : float, optional
        Power law index for high-velocity extrapolation. Default is 4.0.
    v_ref: reference velocity at which to normalize cross section strengtht to "amp_at_vref"

    Returns
    -------
    CrossSectionInterpolator
    """
    if v is None:
        v = np.logspace(-1, 4, 100)
    if potential == 'MIXTURE':
        cross_section_interpolator_att, _, _, _ = make_cross_section_interpolator('ATTRACTIVE_YUKAWA',
                                                                                  cross_section_type,
                                                                                  v, v_power, v_ref)
        cross_section_interpolator_rep, _, _, _ = make_cross_section_interpolator('REPULSIVE_YUKAWA',
                                                                                  cross_section_type,
                                                                                  v, v_power, v_ref)
        return (MixtureCrossSectionInterpolator([cross_section_interpolator_att,
                                                 cross_section_interpolator_rep]),
                None,
                cross_section_interpolator_att.log10_mass_ratio_values,
                cross_section_interpolator_att.log10alpha_values)
    else:
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
