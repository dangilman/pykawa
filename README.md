# pykawa
[![CI](https://github.com/dangilman/pykawa/actions/workflows/ci.yml/badge.svg)](https://github.com/dangilman/pykawa/actions/workflows/ci.yml)

<p align="center">
  <img src="pykawa_logo.png" width="600"/>
</p>

The phase shifts and scattering cross sections for Yukawa potentials. 

Solutions are obtained following the variable phase approach. The mathematica scripts used to compute the cross sections can be found in the "mathematica_scripts" directory. 

## Features

See the notebooks for example usage. 
- Load precomputed solutions for the scattering cross section, including the angle-dependent, total, momentum, and viscosity cross sections, for Yukawa potentials in the Born and semi-classical regime. Solutions include both repulsive and attractive (resonant) cross sections
- Use interpolators to obtain solutions across a range of parameter space
- Compute thermally averaged cross sections relevant for structure formation

## Author

pykawa was created in 2026 by Daniel Gilman.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.

## Attribution 
When using methods in this package, please provide a link to this repo and cite Gilman et al. (2026), arXiv:2610.40766. 

The SIDM model considered by Gilman et al. (2026) is stored as the FERMION_CHARGED_SCALAR in the cross_section_interpolator module, and was computed following methods presented by Flores, Petraki, Pignard (2026). When using this model, please cite: 
M. M. Flores, K. Petraki, and T. Pignard, Variable-phase methods and data for long-range scattering, arXiv:2610.30261.

and also cite the dataset:

M. M. Flores, K. Petraki, and T. Pignard, YukawaVPS data: phase shifts, Sommerfeld factors, and cross sections for Yukawa scattering, Zenodo (2026), doi:10.5281/zenodo.23475703.

For the other Yukawa potential models, phase shifts are computed with the variable-phase method (Calogero 1967), using the formulation of Chu, Garcia-Cely & Murayama (2020), arXiv:1908.06067.
