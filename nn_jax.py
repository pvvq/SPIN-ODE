import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import diffrax
from torch.utils.data import default_collate
from einops import rearrange

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

def numpy_collate(batch):
    return jax.tree_util.tree_map(np.asarray, default_collate(batch))

def jax_collate(batch):
    return jax.tree_util.tree_map(jnp.array, default_collate(batch))

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

    def get_k(self, *args, **kargs):
        return None

class CRNN(nnx.Module):
    def __init__(self, num_spc, num_react, coef_in, coef_out, RO2_IDX=None, RO2_K_IDX=None, k=None):
        super().__init__()
        self.num_spc = num_spc
        self.num_react = num_react

        if k is None:
            key = jax.random.key(42)
            k = jnp.exp(jax.random.normal(key, (self.num_react)))

        self.ln_k = nnx.Param(jnp.log(k).reshape(-1, 1))

        self.coef_in = Var(jnp.array(coef_in))
        self.coef_out = Var(jnp.array(coef_out))
        self.RO2_IDX = Var(jnp.array(RO2_IDX)) if RO2_IDX is not None else None
        self.RO2_K_IDX = Var(jnp.array(RO2_K_IDX)) if RO2_K_IDX is not None else None

    def __call__(self, t, x):
        # x: [B, n_spc]
        x = jnp.clip(x, 1e-30, 1e30)

        poly = self.coef_in.value.T @ jnp.log(x).reshape(-1, self.num_spc, 1)  # [B, n_react, 1]

        if self.RO2_IDX is not None:  # for toy-44 only, get sum of RO2 species
            RO2_val = x[:, self.RO2_IDX.value].sum(axis=1, keepdims=True).reshape(-1,1,1)  # [B,1,1]
            RO2_mat = jnp.zeros((x.shape[0], self.num_react, 1))
            RO2_mat = RO2_mat.at[:, self.RO2_K_IDX.value, :].set(jnp.log(RO2_val))  # [B, n_react, 1]
            poly += RO2_mat  # RO2-dependent rate for toy-44

        rate = jnp.exp(jnp.expand_dims(self.ln_k.value, 0).repeat(x.shape[0], axis=0) + poly)  # [B, n_react, 1]
        x_out = self.coef_out.value @ rate  # [B, n_spc, 1]

        return jnp.squeeze(x_out)  # [B, n_spc]

    def get_k(self, *args, **kargs):
        return jnp.exp(self.ln_k).reshape(-1)

class NeuralODE(nnx.Module):
    def __init__(self, ode):
        super().__init__()
        self.ode = ode  # expects (t, y) → dy, where y is [B, dim]
        
        self.stiff_solver = diffrax.Kvaerno3()
        self.euler_solver = diffrax.Euler()
        self.adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=8192)
        self.adaptive_step_crtl = diffrax.PIDController(rtol=1e-6, atol=1e-3)
        self.fix_step_crtl = diffrax.ConstantStepSize()

    def __call__(self, init_conc, time):
        """
        Args:
            init_conc: [B, ...]
            time: [T]
        Returns:
            [B, T, ...]
        """
        def _batched_rhs(t, y, args):
            return self.ode(t, y)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(_batched_rhs),
            self.stiff_solver,
            t0=time[0],
            t1=time[-1],
            dt0=0.0002,
            y0=init_conc,
            saveat=diffrax.SaveAt(ts=time),
            adjoint=self.adjoint,
            max_steps=8192,
            throw=False,
            stepsize_controller=self.adaptive_step_crtl,
        )
        # set EQX_ON_ERROR=nan to stop runtime error
        # if sol.result != diffrax.RESULTS.successful:
        #     jax.debug.print("rt error, fall back to fix-step method")
        # sol = diffrax.diffeqsolve(
        #     diffrax.ODETerm(_batched_rhs),
        #     self.euler_solver,
        #     t0=time[0],
        #     t1=time[-1],
        #     dt0=0.1,
        #     y0=init_conc,  # [B, dim]
        #     saveat=diffrax.SaveAt(ts=time),
        #     adjoint=self.adjoint,
        #     max_steps=8192,
        #     stepsize_controller=self.fix_step_crtl,
        # )
        ret = rearrange(sol.ys, 't b s -> b t s') #.clip(min=1e-30)

        return ret  # shape: [B, T, ...]

    def get_k(self, *args, **kargs):
        return self.ode.get_k(*args, **kargs)
