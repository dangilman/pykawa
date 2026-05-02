c_kms = 299729.0      # speed of light in km/s
hbarc  = 0.197327e-13 # GeV·cm
mchigram = 1.0 * 1.78e-24  # g
k_GeV_per_v = 1 / (2 * c_kms)   # GeV
# Convert to cm^-1 for consistency check (optional)
#
# # Scattering amplitude [GeV^-1]
# f_theta = np.zeros(len(theta), dtype=complex)
# for l in range(lconv + 1):
#     dl = ph[l]
#     Pl = eval_legendre(l, cos_theta)
#     f_theta += (2*l + 1) * np.exp(1j*dl) * np.sin(dl) * Pl
# f_theta /= k_GeV                  # GeV^-1
#
# # Differential cross section [GeV^-2 -> cm^2 -> cm^2/g]
# dsigma = np.abs(f_theta)**2       # GeV^-2
# dsigma *= hbarc**2                # cm^2
# dsigma /= mchigram                # cm^2/g/sr
