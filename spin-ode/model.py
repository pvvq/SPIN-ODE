from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx


# Kim, Suyong, et al. "Stiff neural ordinary differential equations."
class ScaleMLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        key,
    ):
        """
        Args:
            num_spc: number of species
            scale: time series scale statistics
        """
        super().__init__()

        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(self, y: jax.Array, scale: dict) -> jax.Array:
        y_safe = jnp.clip(y, 0.0)
        y_scaled = y_safe / scale["yScale"]
        dy_dt = self.mlp(y_scaled) * scale["ytScale"]
        return dy_dt


def neural_ode(t, y, args):
    nn = args["neural_network"]
    return nn(y, args["scale"])


def kinetic_ode(t, y, args):
    """time derivative of model state"""
    ro2_sum = args["update_RO2_fn"](y)
    k = args["update_rconst_RO2_fn"](args["k_static"], args["ro2_coef"], ro2_sum)
    dy_dt = args["rate_law_fn"](y, k, args["stoicm"])
    return dy_dt


def solve(params: dict, ts, y0, ode: Callable):
    """
    Solve the model ODE in time from initial state

    Args:
        params: model parameters
        ts: time span
        y0: initial state
        ode: function of time derivative
    Returns:
        integrated model state
    """
    solver_cfg = params["solver"]

    sol = dfx.diffeqsolve(
        dfx.ODETerm(ode),
        dfx.Kvaerno5(),
        t0=ts[0],
        t1=ts[-1],
        y0=y0,
        saveat=dfx.SaveAt(ts=ts),
        dt0=None,
        max_steps=solver_cfg["max_steps"],
        adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=solver_cfg["checkpoints"]),
        stepsize_controller=dfx.PIDController(
            rtol=solver_cfg["rtol"], atol=solver_cfg["atol"]
        ),
        throw=True,
        args=params,
    )
    return sol.ys
