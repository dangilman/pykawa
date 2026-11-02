"""Even and odd parts of the viscosity cross section of the attractive / repulsive Yukawa potential, interpolated in
(xi, zeta) from exact numerical solutions of the Schroedinger equation.

    from yukawa_viscosity_even_odd import yukawa_viscosity_even_odd
    S_even, S_odd = yukawa_viscosity_even_odd(xi, zeta, sign=+1)   # +1 attractive, -1 repulsive (numpy arrays)

    zeta = alpha / v_rel,  xi = mu alpha / m_phi  (mu = reduced mass = m_chi/2),  kappa = xi/zeta,  beta = zeta^2/xi.

Output in units of 4 pi / k^2 (k = mu v_rel), for distinguishable particles:
    S_even = sum_{l even} r_l sin^2(delta_{l+2} - delta_l),  S_odd = same over odd l,  r_l = (l+1)(l+2)/(2l+3),
    sigma_V(distinguishable) = 4 pi / k^2 (S_even + S_odd);  identical particles: 8 pi / k^2 S_even (or S_odd).
See README.md for the domain and the accuracy.
"""
import glob
import os

import numpy as np
from scipy.special import k0e, k1e, kve

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_NSUB, _NCOL = 3400, 379                 # refined table: rows every 1/3400 dex in xi, 379 zeta columns per row
_REFINED_AXES = {"log10_xi": 0, "j0": 0, "Delta": 1, "R_even": 0, "R_odd": 0}   # xi-row axis of each array
_r = lambda l: (l + 1.0) * (l + 2.0) / (2 * l + 3.0)


def save_chunks(path, n_chunks, axes=_REFINED_AXES):
    """Split the npz at path into path_0.npz, path_1.npz, ... along the xi-row axis of each array (to keep each file
    under GitHub's size limit).  load_chunks reassembles them exactly."""
    stem = path[:-4]
    with np.load(path) as d:
        pieces = {k: np.array_split(d[k], n_chunks, axis=a) for k, a in axes.items()}
    for i in range(n_chunks):
        np.savez_compressed(f"{stem}_{i}.npz", **{k: pieces[k][i] for k in axes})


def load_chunks(stem, axes=_REFINED_AXES):
    """Arrays saved by save_chunks(stem + '.npz', ...), concatenated in chunk order."""
    files = sorted(glob.glob(f"{stem}_*.npz"), key=lambda f: int(f[len(stem) + 1:-4]))
    if not files:
        raise FileNotFoundError(f"no table chunks {stem}_*.npz")
    parts = []
    for f in files:
        with np.load(f) as p:
            parts.append({k: p[k] for k in axes})
    return {k: np.concatenate([p[k] for p in parts], axis=a) for k, a in axes.items()}


def _classics(kappa, beta, lmin, attractive):
    """sigma_V m_phi^2 / pi of arXiv:2011.04679 (CLASSICS), with the lower limit lmin (1/2 even, 3/2 odd)."""
    out = np.empty(kappa.shape)
    s = beta < 0.5
    b2, k = 2 * beta[s], kappa[s]
    lm = np.maximum(lmin, b2 * k)
    y = lm / k
    eta = y ** 2 * np.exp(-2 * y) * (k0e(y) * kve(2, y) - k1e(y) ** 2)
    Z = (lm ** 2 - lmin ** 2) / (2 * k ** 2 * b2 ** 2) + eta
    out[s] = 4 * beta[s] ** 2 * Z * np.exp((0.67 if attractive else -0.370562) * (np.maximum(beta[s], 0.1) - 0.1))
    m = (beta >= 0.5) & (beta <= 25)
    out[m] = 2.5 * np.log(beta[m] + 1.05) if attractive else 2.8 * np.log(beta[m] + 0.80)
    h = beta > 25
    if attractive:
        lb = np.log(beta[h])
        out[h] = (1 + lb - 1 / (2 * lb)) ** 2 / 2
    else:
        lv = (9 - np.cos(4) - 4 * np.sin(4)) / 16
        l2b = np.log(2 * beta[h])
        out[h] = l2b * (lv * l2b - (2 * lv - 1) * np.log(l2b))
    return out


class _Tables:
    """Priority of the sources at each point:
    1  attractive only, 0.79 <= xi <= 2500, xi/10 <= zeta <= ~600 xi: table refined in xi (threshold resonances);
    2  0.01-dex lattice of exact values (log10 xi = -6 + 0.01 i, log10 zeta = -4 + 0.01 j);
    3  CLASSICS even/odd formulas where kappa > 10 and there are no data;
    NaN otherwise.  return_source=True also returns the source used (1, 2, 3, or 0 for NaN)."""

    def __init__(self, data=_DATA):
        L = np.load(os.path.join(data, "parity_lattice.npz"))
        self.lat = {(s, p): L[f"S_{p}_{s}"] for s in "AR" for p in ("even", "odd")}
        R = load_chunks(os.path.join(data, "refined_A"))     # refined_A_0.npz, refined_A_1.npz, ...
        self.k0 = int(round(R["log10_xi"][0] * _NSUB))
        self.j0 = R["j0"]
        self.Delta = R["Delta"].astype(float)        # delta_2 - delta_0, delta_3 - delta_1 on the refined rows
        self.RE, self.RO = R["R_even"].astype(float), R["R_odd"].astype(float)   # rest l >= 2

    def _refined(self, lx, lz):
        fk = lx * _NSUB - self.k0
        i = np.floor(fk).astype(int); tx = fk - i
        fj = (lz + 4) * 100
        jg = np.floor(fj).astype(int); tz = fj - jg
        ok = (i >= 0) & (i < self.j0.size - 1)
        i = np.clip(i, 0, self.j0.size - 2)
        c0, c1 = jg - self.j0[i], jg - self.j0[i + 1]
        ok &= (c0 >= 0) & (c0 < _NCOL - 1) & (c1 >= 0) & (c1 < _NCOL - 1)
        c0, c1 = np.clip(c0, 0, _NCOL - 2), np.clip(c1, 0, _NCOL - 2)
        cor = ((i, c0), (i + 1, c1), (i, c0 + 1), (i + 1, c1 + 1))
        w = np.array([(1 - tx) * (1 - tz), tx * (1 - tz), (1 - tx) * tz, tx * tz])
        S = []
        for par, Rr in ((0, self.RE), (1, self.RO)):
            c = [Rr[a, b] for a, b in cor]                       # rest: bilinear in log (linear if a corner is 0)
            pos = np.all([v > 0 for v in c], axis=0)
            rest = np.where(pos, np.exp(sum(w[k] * np.log(np.where(pos, c[k], 1.0)) for k in range(4))),
                            sum(w[k] * c[k] for k in range(4)))
            d = np.array([self.Delta[par][a, b] for a, b in cor])  # s / p wave: phase difference, unwrapped mod pi
            d[1:] -= np.pi * np.round((d[1:] - d[0]) / np.pi)
            S.append(rest + _r(par) * np.sin(np.sum(w * d, axis=0)) ** 2)
        ok &= np.isfinite(S[0]) & np.isfinite(S[1]) & (S[0] > 0) & (S[1] > 0)
        return S[0], S[1], ok

    def _lattice(self, T, lx, lz):
        fi, fj = (lx + 6) * 100, (lz + 4) * 100
        i, j = np.floor(fi).astype(int), np.floor(fj).astype(int)
        tx, tz = fi - i, fj - j
        ok = (i >= 0) & (i < 1100) & (j >= 0) & (j < 1100)
        i, j = np.clip(i, 0, 1099), np.clip(j, 0, 1099)
        S = []
        for p in ("even", "odd"):
            A = self.lat[(T, p)]
            c = [A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1]]
            ok &= np.all([np.isfinite(v) & (v > 0) for v in c], axis=0)
            lg = [np.log(np.where(np.isfinite(v) & (v > 0), v, 1.0)) for v in c]
            S.append(np.exp((1 - tx) * (1 - tz) * lg[0] + tx * (1 - tz) * lg[1] + (1 - tx) * tz * lg[2] + tx * tz * lg[3]))
        return S[0], S[1], ok

    def __call__(self, xi, zeta, sign=1, return_source=False):
        if sign not in (1, -1):
            raise ValueError("sign must be +1 (attractive) or -1 (repulsive)")
        xi, zeta = np.broadcast_arrays(np.asarray(xi, float), np.abs(np.asarray(zeta, float)))
        shp = xi.shape
        x, z = xi.ravel(), zeta.ravel()
        lx, lz = np.log10(x), np.log10(z)
        SE, SO = np.full(x.size, np.nan), np.full(x.size, np.nan)
        src = np.zeros(x.size, int)
        steps = [(1, self._refined)] if sign > 0 else []
        steps.append((2, lambda a, b: self._lattice("A" if sign > 0 else "R", a, b)))
        for tag, f in steps:
            todo = src == 0
            e, o, ok = f(lx, lz)
            ok &= todo
            SE[ok], SO[ok], src[ok] = e[ok], o[ok], tag
        kap, beta = x / z, z * z / x
        cl = (src == 0) & (kap > 10)
        if cl.any():
            SE[cl] = _classics(kap[cl], beta[cl], 0.5, sign > 0) * kap[cl] ** 2 / 8
            SO[cl] = _classics(kap[cl], beta[cl], 1.5, sign > 0) * kap[cl] ** 2 / 8
            src[cl] = 3
        out = (SE.reshape(shp), SO.reshape(shp))
        return out + (src.reshape(shp),) if return_source else out


_tables = None


def yukawa_viscosity_even_odd(xi, zeta, sign=1, return_source=False):
    """(S_even, S_odd) at (xi, zeta) for the attractive (sign=+1) or repulsive (sign=-1) Yukawa potential, in units of
    4 pi / k^2 (distinguishable particles).  The tables in data/ are loaded at the first call (< 1 s, 0.1 GB).
    return_source=True also returns the source used at each point: 1 refined table, 2 0.01-dex table, 3 CLASSICS,
    0 NaN (outside the computed range)."""
    global _tables
    if _tables is None:
        _tables = _Tables()
    return _tables(xi, zeta, sign, return_source)
