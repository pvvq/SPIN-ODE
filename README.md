# Differentiable modelling of chemical kinetic reactions.

## SPIN-ODE: **S**tiff **P**hysics-**I**nformed **N**eural **ODE** for Chemical Reaction Rate Estimation
Code for paper [SPIN-ODE](https://doi.org/10.3233%2FFAIA251040)
    - Version used in the paper: [v1.0.0](https://github.com/pvvq/SPIN-ODE/releases/tag/v1.0.0)
    - More details in [SPIN-ODE README](spin_ode/README.md)

Code for paper [Evaluation of optimisation and Bayesian inference methods for reaction rates in atmospheric chemical mechanisms](https://doi.org/10.48550/arXiv.2609.14569)
    - Version used in this paper [v1.1.1](https://github.com/pvvq/SPIN-ODE/releases/tag/v1.1.1)
    - Mass specturm version: [v1.1.1-mass](https://github.com/pvvq/SPIN-ODE/releases/tag/v1.1.1-mass)

## Environment
Clone with submodules (this pulls in [KPPax](https://github.com/pvvq/KPPax) at the pinned version):
```sh
git clone --recurse-submodules https://github.com/pvvq/SPIN-ODE.git
```
If you already cloned without `--recurse-submodules`:
```sh
git submodule update --init
```

Install the Python dependencies:
```sh
pip install -e .
```

## Acknowledgements
- [Diffrax](https://docs.kidger.site/diffrax/citation/)
- [CRNN](https://github.com/DENG-MIT/CRNN)
- [Stiff Neural ODE](https://github.com/DENG-MIT/StiffNeuralODE)
- [Collocation training for stiff neural ODE](https://github.com/Xiangjun-Huang/training_stiff_NODE_in_WW_modelling)

# License
This project is licenced under GNU GPLv3, see LICENSE for details.

# Change log
## v1.1.1
- update README

## v1.1.0
- Major refactor, more JAX-ish.
- Switch from Flax to Equinox as the nueral netowrk framework.
- Use the KPPax as the kinetic reaction equation parse.

## v1.0.0
- Initial release for the SPIN-ODE paper.
