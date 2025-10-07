from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from flax import nnx
import diffrax


def SMSPELoss(pred, truth):
    """Symmetric Mean Squared Percentage Error"""
    diff = jnp.sqrt(jnp.square(pred - truth))
    scale = jnp.sqrt(jnp.square(pred)) + jnp.sqrt(jnp.square(truth))
    return jnp.log(jnp.mean(diff / scale))

def signed_softlog(x, alpha=10.0):
    return jnp.sign(x) * jnp.log1p(alpha * jnp.abs(x))

def signed_softlog_mse(y_true, y_pred, alpha=10.0):
    y_true_slog = signed_softlog(y_true, alpha)
    y_pred_slog = signed_softlog(y_pred, alpha)
    return jnp.mean((y_pred_slog - y_true_slog) ** 2)

def LogMAELoss(pred, truth):
    return jnp.mean(jnp.abs(jnp.log(pred) - jnp.log(truth)))

def LogMSELoss(pred, truth):
    return jnp.mean(jnp.square(jnp.log(pred) - jnp.log(truth)))

def ScaleMSELoss(pred, truth, yscale):
    scaled_diff = (pred - truth) / yscale
    return jnp.mean(jnp.square(scaled_diff))

def ScaleMAELoss(pred, truth, yscale):
    scaled_diff = (pred - truth) / yscale
    return jnp.mean(jnp.abs(scaled_diff))

def TVLoss1D(x, scale, window_size=1):
    diff = x[window_size:, ...] - x[:-window_size, ...]
    tv = jnp.square(diff)

    loss = jnp.mean(tv / scale)
    return loss


class Var(nnx.Variable):
    """Non-trainable variable"""
    pass

class ScaleMLP(nnx.Module):
    def __init__(
            self,
            num_spc: int,
            scale: dict[str, ArrayLike],
            hidden_size: int = 32,
            *,
            rngs: nnx.Rngs
        ):
        """
        Args:
            num_spc: number of species
            scale: dict of concentration and time scales
            hidden_size: hidden layer size
        """
        super().__init__()
        self.num_spc = num_spc

        self.linear1 = nnx.Linear(self.num_spc, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_size, self.num_spc, rngs=rngs)

        self.yMin = Var(scale['yMin'])
        self.yscale = Var(scale['yScale'])
        self.ytscale = Var(scale['ytScale'])

    def __call__(self, t: jax.Array, c: jax.Array) -> jax.Array:
        c = (c - self.yMin.value) / self.yscale.value
        x = nnx.gelu(self.linear1(c))
        x = nnx.gelu(self.linear2(x))
        dcdt = self.linear3(x)
        dcdt = dcdt * self.ytscale.value
        return dcdt


class PowerRateLaw(nnx.Module):
    def __init__(
            self,
            stoi_reac: ArrayLike,
            stoi_prod: ArrayLike,
            RO2_IDX: ArrayLike,
            RO2_K_IDX: ArrayLike,
            k: ArrayLike,
        ):
        """
        Args:
            stoi_reat: reactant stoichiometric coefficients
            stoi_prod: product stoichiometric coefficients,
            RO2_IDX: index of RO2 species,
            RO2_K_IDX: index of RO2-dependent rate coefficients,
            k: rate coefficients (learnable),
        """
        super().__init__()
        
        self.stoi_reat = Var(jnp.asarray(stoi_reac))
        self.stoi_net = Var(jnp.asarray(stoi_prod - stoi_reac))
        self.RO2_IDX = Var(jnp.asarray(RO2_IDX, dtype=jnp.int32))
        self.RO2_K_IDX = Var(jnp.asarray(RO2_K_IDX, dtype=jnp.int32))

        self.log_k = nnx.Param(jnp.log(jnp.asarray(k)))

    def __call__(self, t: jax.Array, y: jax.Array) -> jax.Array:
        """
        Args:
            t: current time
            y: current concentration vector
        Returns:
            time derivative of concentrations
        """
        rate = jnp.exp(self.log_k.value) * jnp.prod(y[:, None] ** self.stoi_reat.value, axis=0)
        
        RO2 = jnp.sum(y[self.RO2_IDX.value])
        rate = rate.at[self.RO2_K_IDX.value].multiply(RO2, unique_indices=True)
        
        dy_dt  = self.stoi_net.value @ rate
        return dy_dt
    
    def get_k(self) -> jax.Array:
        """Returns learnable rate coefficients"""
        return jnp.exp(self.log_k.value)
    
class LogRateLaw(nnx.Module):
    def __init__(
            self,
            stoi_reac: ArrayLike,
            stoi_prod: ArrayLike,
            RO2_IDX: ArrayLike,
            RO2_K_IDX: ArrayLike,
            k: ArrayLike,
        ):
        """
        Args:
            stoi_reat: reactant stoichiometric coefficients
            stoi_prod: product stoichiometric coefficients,
            RO2_IDX: index of RO2 species,
            RO2_K_IDX: index of RO2-dependent rate coefficients,
            k: rate coefficients (learnable),
        """
        super().__init__()
        
        self.stoi_reat = Var(jnp.asarray(stoi_reac))
        self.stoi_net = Var(jnp.asarray(stoi_prod - stoi_reac))
        self.RO2_IDX = Var(jnp.asarray(RO2_IDX, dtype=jnp.int32))
        self.RO2_K_IDX = Var(jnp.asarray(RO2_K_IDX, dtype=jnp.int32))

        self.log_k = nnx.Param(jnp.log(jnp.asarray(k)))

    def __call__(self, t: jax.Array, y: jax.Array) -> jax.Array:
        """
        Args:
            t: current time
            y: current concentration vector
        Returns:
            time derivative of concentrations
        """
        log_y = jnp.log(jnp.clip(y, 1e-30, 1e30))
        rate = jnp.exp(self.log_k.value) * jnp.exp(log_y @ self.stoi_reat.value)
        
        RO2 = jnp.sum(y[self.RO2_IDX.value])
        rate = rate.at[self.RO2_K_IDX.value].multiply(RO2, unique_indices=True)

        dy_dt = self.stoi_net.value @ rate
        return dy_dt
    
    def get_k(self) -> jax.Array:
        """Returns learnable rate coefficients"""
        return jnp.exp(self.log_k.value)

def ode_solver(
        *,
        solver = diffrax.Kvaerno3(),
        dt0 = None,
        adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=8192),
        max_steps: int = 8192,
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-7),
        throw: bool = False,
    ) -> Callable[[Callable, jax.Array, jax.Array], jax.Array]:
    """
    Args:
        ode: ode function
        ts: time span
        y0: initial state
    Returns:
        solved states in saveat time
    """
    def ode_solve(
            ode: Callable,
            ts: jax.Array, 
            y0: jax.Array,
        ) -> jax.Array:
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: ode(t, y)),
            solver,
            t0=ts[0],
            t1=ts[-1],
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            dt0=dt0,
            adjoint=adjoint,
            max_steps=max_steps,
            stepsize_controller=stepsize_controller,
            throw=throw,
        )
        return sol.ys
    
    return ode_solve