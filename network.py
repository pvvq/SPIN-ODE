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
    # x: [B, t, s]
    diff = x[:, window_size:, :] - x[:, :-window_size, :]
    tv = jnp.square(diff)

    loss = jnp.mean(tv / scale)
    return loss

def gradient(x: jnp.ndarray, t: jnp.ndarray = None) -> jnp.ndarray:
    """
    Compute finite difference gradient along time axis (dim=1) for x of shape [B, T, D].
    If t is None, assumes uniform spacing of 1.0 along time.
    
    Args:
        x: Tensor of shape [B, T, D]
        t: Optional 1D array of shape [T]; if None, assumes t = [0, 1, ..., T-1]
    Returns:
        Gradient dx/dt of shape [B, T, D]
    """
    B, T, D = x.shape
    if t is None:
        t = jnp.arange(T, dtype=x.dtype)

    def grad_single_batch(xi):  # xi: [T, D]
        grad = jnp.zeros_like(xi)

        # Forward difference at start
        grad = grad.at[0].set((xi[1] - xi[0]) / (t[1] - t[0]))

        # Backward difference at end
        grad = grad.at[-1].set((xi[-1] - xi[-2]) / (t[-1] - t[-2]))

        # Central difference for interior
        dt = t[2:] - t[:-2]          # [T-2]
        dx = xi[2:] - xi[:-2]        # [T-2, D]
        grad = grad.at[1:-1].set(dx / dt[:, None])

        return grad

    return jax.vmap(grad_single_batch)(x)  # apply across batch dimension


class Var(nnx.Variable):
    pass

class ScaleMLP(nnx.Module):
    def __init__(self, num_spc, num_react, scale, hidden_size=32, *, rngs: nnx.Rngs):
        super().__init__()
        self.num_spc = num_spc
        self.num_react = num_react

        self.linear1 = nnx.Linear(self.num_spc, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_size, self.num_spc, rngs=rngs)

        self.yMin = Var(scale['yMin'])
        self.yscale = Var(scale['yScale'])
        self.ytscale = Var(scale['ytScale'])

    def __call__(self, t, c):
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
            k_init: ArrayLike,
        ):
        """
        Args:
            stoi_reat: reactant stoichiometric coefficients
            stoi_prod: product stoichiometric coefficients,
            RO2_IDX: index of RO2 species,
            RO2_K_IDX: index of RO2-dependent rate coefficients,
            k_init: initial rate coefficients (learnable),
        """
        super().__init__()
        
        self.stoi_reat = jnp.asarray(stoi_reac)
        self.stoi_net = jnp.asarray(stoi_prod - stoi_reac)
        self.RO2_IDX = jnp.asarray(RO2_IDX, dtype=jnp.int32)
        self.RO2_K_IDX = jnp.asarray(RO2_K_IDX, dtype=jnp.int32)

        self.log_k = nnx.Param(jnp.log(jnp.asarray(k_init)))

    def __call__(self, t: jax.Array, y: jax.Array) -> jax.Array:
        """
        Args:
            t: current time
            y: current concentration vector
        Returns:
            time derivative of concentrations
        """
        rate = jnp.exp(self.log_k.value) * jnp.prod(y[:, None] ** self.stoi_reat, axis=0)
        
        RO2 = jnp.sum(y[self.RO2_IDX])
        rate = rate.at[self.RO2_K_IDX].multiply(RO2)
        
        dy_dt  = self.stoi_net @ rate
        return dy_dt
    
class LogRateLaw(nnx.Module):
    def __init__(
            self,
            stoi_reac: ArrayLike,
            stoi_prod: ArrayLike,
            RO2_IDX: ArrayLike,
            RO2_K_IDX: ArrayLike,
            k_init: ArrayLike,
        ):
        """
        Args:
            stoi_reat: reactant stoichiometric coefficients
            stoi_prod: product stoichiometric coefficients,
            RO2_IDX: index of RO2 species,
            RO2_K_IDX: index of RO2-dependent rate coefficients,
            k_init: initial rate coefficients (learnable),
        """
        super().__init__()
        
        self.stoi_reat = jnp.asarray(stoi_reac)
        self.stoi_net = jnp.asarray(stoi_prod - stoi_reac)
        self.RO2_IDX = jnp.asarray(RO2_IDX, dtype=jnp.int32)
        self.RO2_K_IDX = jnp.asarray(RO2_K_IDX, dtype=jnp.int32)

        self.log_k = nnx.Param(jnp.log(jnp.asarray(k_init)))

    def __call__(self, t: jax.Array, y: jax.Array) -> jax.Array:
        """
        Args:
            t: current time
            y: current concentration vector
        Returns:
            time derivative of concentrations
        """
        log_y = jnp.log(jnp.clip(y, 1e-30, 1e30))
        rate = jnp.exp(self.log_k.value) * jnp.exp(log_y @ self.stoi_reat)
        
        RO2 = jnp.sum(y[self.RO2_IDX])
        rate = rate.at[self.RO2_K_IDX].multiply(RO2)

        dy_dt = self.stoi_net @ rate
        return dy_dt

class Solverax(nnx.Module):
    """Wrapper of diffrax ODE solver"""
    def __init__(self, ode: Callable):
        super().__init__()
        self.ode = ode

    def _rhs(self, t, y, args):
        return self.ode(t, y)

    def __call__(self, ts: jax.Array, y0: jax.Array) -> jax.Array:
        """
        Args:
            ts: time span
            y0: initial state
        Returns:
            solved states in saveat time
        """
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self._rhs),
            diffrax.Kvaerno3(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.0002,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=8192),
            max_steps=8192,
            throw=False,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-7),
        )
        return sol.ys