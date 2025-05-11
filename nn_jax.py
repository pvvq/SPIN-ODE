import jax
import jax.numpy as jnp
from flax import nnx
import diffrax
from torch.utils.data import default_collate
from einops import rearrange
from scipy.optimize import nnls

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

    # def get_k(self, c, coef_in, coef_out, RO2_IDX=None, RO2_K_IDX=None):
    #     """
    #     c: a series of concentration [B, num_spc]
    #     """
    #     poly = jnp.exp(
    #         jnp.einsum('rs,bs->br',
    #                 jnp.array(coef_in).T,
    #                 jnp.log(c))
    #     )  # polynominal term in the rate law
    #     if RO2_IDX is not None:
    #         RO2_val = c[:, jnp.array(RO2_IDX)].sum(axis=1, keepdims=True)  # [B,1]
    #         RO2_mat = jnp.zeros((c.shape[0], self.num_react))
    #         RO2_mat = RO2_mat.at[:, jnp.array(RO2_K_IDX)].set(RO2_val)  # [B, n_react]
    #         poly = poly * RO2_mat  # some rate coef are RO2 conc dependent

    #     dcdt = self.__call__(None, c)
    #     coef_out_pinv = jnp.linalg.pinv(jnp.array(coef_out))
    #     rate = jnp.einsum('rs,bs->br', coef_out_pinv, dcdt)
    #     sol = jnp.linalg.lstsq(poly, rate)
    #     pred_k = sol[0]
    #     return pred_k

class LogMLP(nnx.Module):
    def __init__(self, num_spc, num_react, coef_out, hidden_size=32, *, rngs: nnx.Rngs):
        super().__init__()
        self.num_spc = num_spc
        self.num_react = num_react
        self.coef_out = Var(jnp.array(coef_out))

        self.linear1 = nnx.Linear(self.num_spc, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_size, self.num_react, rngs=rngs)

    def rate(self, t, c):
        log_c = jnp.log(c)
        x = nnx.gelu(self.linear1(log_c))
        x = nnx.gelu(self.linear2(x))
        log_r = self.linear3(x)
        r = jnp.exp(log_r)
        return r

    def __call__(self, t, c):
        r = self.rate(t, c)
        dc_dt = jnp.einsum('sr,br->bs', self.coef_out.value, r)
        return dc_dt

    # def get_k(self, c, coef_in, coef_out, RO2_IDX=None, RO2_K_IDX=None):
    #     poly = jnp.exp(
    #         jnp.einsum('rs,bs->br',
    #                 jnp.array(coef_in).T,
    #                 jnp.log(c))
    #     )  # polynominal term in the rate law
    #     if RO2_IDX is not None:
    #         RO2_val = c[:, jnp.array(RO2_IDX)].sum(axis=1, keepdims=True)  # [B,1]
    #         RO2_mat = jnp.zeros((c.shape[0], self.num_react))
    #         RO2_mat = RO2_mat.at[:, jnp.array(RO2_K_IDX)].set(RO2_val)  # [B, n_react]
    #         poly = poly * RO2_mat  # some rate coef are RO2 conc dependent

    #     rate = model.rate(None, c)
    #     sol = jnp.linalg.lstsq(poly, rate)
    #     pred_k = sol[0]
    #     return pred_k

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

def get_k_linreg(ode, c, coef_in, coef_out, RO2_IDX=None, RO2_K_IDX=None):
    """
    c: a series of concentration [B, num_spc]
    # """
    monomial = jnp.einsum('rs,bs->br', jnp.array(coef_in).T, jnp.log(c)) # monomial term in the rate law
    print("prod1", monomial[1])
    if RO2_IDX is not None:
        RO2_val = c[:, jnp.array(RO2_IDX)].sum(axis=1, keepdims=True)  # [B,1]
        RO2_mat = jnp.ones((c.shape[0], self.num_react))
        RO2_mat = RO2_mat.at[:, jnp.array(RO2_K_IDX)].set(RO2_val)  # [B, n_react]
        monomial = monomial * RO2_mat  # some rate coef are RO2 conc dependent
        print("RO2_mat", RO2_mat[1])
    print("prod2", monomial[1])
    dcdt = ode(None, c)
    print("dcdt", dcdt[1])

    batch_size = dcdt.shape[0]

    # Construct A_k: vertically stack each S * monomial[i]
    A_k = jnp.concatenate([(coef_out * monomial[i]) for i in range(batch_size)], axis=0)   # shape [batch * n_species, n_reactions]

    # Flatten dcdt
    b = dcdt.reshape(-1)  # shape [batch * n_species]

    # Convert to numpy for scipy.nnls
    A_k_np = jnp.asarray(A_k).astype(jnp.float64).copy()  # ensure writeable for SciPy
    b_np = jnp.asarray(b).astype(jnp.float64).copy()

    # Solve non-negative least squares
    k_np, rnorm = nnls(A_k_np, b_np)

    # coef_out_pinv = jnp.linalg.pinv(jnp.array(coef_out))
    # rate = jnp.einsum('rs,bs->br', coef_out_pinv, dcdt)
    # pred_k = jnp.sum(monomial * rate, axis=0) / (jnp.sum(monomial ** 2, axis=0))
    return k_np

class Attention(nnx.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., *, rngs: nnx.Rngs):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        # self.dropout = nnx.Dropout(dropout, rngs=rngs)

        self.to_qkv = nnx.Linear(dim, inner_dim * 3, use_bias = False, rngs=rngs)

        self.to_out = nnx.Sequential(
            nnx.Linear(inner_dim, dim, rngs=rngs),
            # nnx.Dropout(dropout)
        )

        self.ffn = nnx.Sequential(
            nnx.LayerNorm(dim, rngs=rngs),
            nnx.Linear(dim, dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(dim, dim, rngs=rngs),
        )

    def forward(self, x):
        re = x
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = q @ k.transpose(-1, -2) * self.scale

        attn = nnx.Softmax(dots, axis=-1)
        # attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = re + self.to_out(out)
        out = out + self.ffn(out)
        return out


class AutoEncoder(nnx.Module):
    def __init__(self, input_x_dim, input_y_dim, output_dim, latent_dims, coef_in, coef_out, guess_k, ode_hidden=8, ode_width=256, *, rngs):
        super().__init__()
        self.latent_dims = latent_dims
        self.output_dim = output_dim
        self.expand_x = nnx.Sequential(
            nnx.Linear(input_x_dim, latent_dims, rngs=rngs)
        )
        self.expand_y = nnx.Sequential(
            nnx.Linear(input_y_dim, latent_dims, rngs=rngs),
            nnx.relu,
        )
        self.sa = Attention(latent_dims, rngs=rngs)
        self.nODE = NeuralODE(lambda t, y: y)

        self.k_initial = Var(jnp.array(guess_k).reshape(-1, 1))
        self.k_scale = nnx.Param(jax.random.normal(jax.random.key(0), (49, 1)))
        self.k_shift = nnx.Param(jax.random.normal(jax.random.key(1), (49, 1)))
        self.s_in = Var(jnp.array(coef_in))
        self.s_out = Var(jnp.array(coef_out))
        self.k_mlp = nnx.Linear(1, latent_dims, rngs=rngs)
        self.decoder = nnx.Sequential(
            nnx.Linear(latent_dims*5, latent_dims, rngs=rngs),
            nnx.Linear(latent_dims, latent_dims, rngs=rngs),
            nnx.Linear(latent_dims, output_dim, rngs=rngs),
        )
    
    def forward(self, initial, query, current_t, future_t):
        self.k = self.k_initial * (1 + self.k_scale) + self.k_shift
        # future_t: B x 5
        bs = initial.shape[0]
        num_chem = initial.shape[1]
        x = self.expand_x(initial)
        y = self.expand_y(query)
        # Encoding
        latent_z = self.sa(x)  # b * 44 * dim
        # kinect rule
        kk = self.k.unsqueeze(0).repeat(bs, 1, 1) # 49 * 1 --> b * 49 * dim
        kk = self.k_mlp(kk)
        latent_z = self.s_in.unsqueeze(0).repeat(bs, 1, 1) @ latent_z  # b * 49 * dim
        latent_z = latent_z * kk
        latent_z = self.s_out.unsqueeze(0).repeat(bs, 1, 1) @ latent_z # b * 44 * dim

        # ODE [:,:,-1]
        z_pred = self.nODE(latent_z, future_t[0])  # b * T * 44 * dim
        z_pred = z_pred.transpose(2,1).reshape(bs, num_chem, -1)  # [bs, num_chem, n_times * latent_dims]

        out = self.decoder(z_pred)

        return (out + initial[:,:,-1].unsqueeze(2).repeat(1, 1, self.output_dim))