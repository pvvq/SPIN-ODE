"""
Jax implementation of the code in paper: [Stiff Neural ODE](https://doi.org/10.1063/5.0060697)

Original code: https://github.com/DENG-MIT/StiffNeuralODE/blob/main/POLLU/POLLU.jl
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import diffrax
from diffrax import ODETerm, SaveAt, PIDController, Kvaerno3
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# ── dirs ──────────────────────────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ── hyper-params ──────────────────────────────────────────────────────────────
is_restart = False
n_epoch = 25000
ntotal = 20
n_plot = 50
grad_max = 1.0e2
batch_size = ntotal

lb = 1.0e-6
t_end = 60.0
tspan = (0.0, t_end)
tsteps = jnp.linspace(0, t_end, ntotal)

# ── rate constants ────────────────────────────────────────────────────────────
k1 = 0.35e0
k2 = 0.266e2
k3 = 0.123e5
k4 = 0.86e-3
k5 = 0.82e-3
k6 = 0.15e5
k7 = 0.13e-3
k8 = 0.24e5
k9 = 0.165e5
k10 = 0.9e4
k11 = 0.22e-1
k12 = 0.12e5
k13 = 0.188e1
k14 = 0.163e5
k15 = 0.48e7
k16 = 0.35e-3
k17 = 0.175e-1
k18 = 0.1e9
k19 = 0.444e12
k20 = 0.124e4
k21 = 0.21e1
k22 = 0.578e1
k23 = 0.474e-1
k24 = 0.178e4
k25 = 0.312e1


# ── pollution ODE ─────────────────────────────────────────────────────────────
def pollu(t, y, args):
    r1 = k1 * y[0]
    r2 = k2 * y[1] * y[3]
    r3 = k3 * y[4] * y[1]
    r4 = k4 * y[6]
    r5 = k5 * y[6]
    r6 = k6 * y[6] * y[5]
    r7 = k7 * y[8]
    r8 = k8 * y[8] * y[5]
    r9 = k9 * y[10] * y[1]
    r10 = k10 * y[10] * y[0]
    r11 = k11 * y[12]
    r12 = k12 * y[9] * y[1]
    r13 = k13 * y[13]
    r14 = k14 * y[0] * y[5]
    r15 = k15 * y[2]
    r16 = k16 * y[3]
    r17 = k17 * y[3]
    r18 = k18 * y[15]
    r19 = k19 * y[15]
    r20 = k20 * y[16] * y[5]
    r21 = k21 * y[18]
    r22 = k22 * y[18]
    r23 = k23 * y[0] * y[3]
    r24 = k24 * y[18] * y[0]
    r25 = k25 * y[19]

    dy = jnp.array(
        [
            -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25,
            -r2 - r3 - r9 - r12 + r1 + r21,
            -r15 + r1 + r17 + r19 + r22,
            -r2 - r16 - r17 - r23 + r15,
            -r3 + r4 + r4 + r6 + r7 + r13 + r20,
            -r6 - r8 - r14 - r20 + r3 + r18 + r18,
            -r4 - r5 - r6 + r13,
            r4 + r5 + r6 + r7,
            -r7 - r8,
            -r12 + r7 + r9,
            -r9 - r10 + r8 + r11,
            r9,
            -r11 + r10,
            -r13 + r12,
            r14,
            -r18 - r19 + r16,
            -r20,
            r20,
            -r21 - r22 - r24 + r23 + r25,
            -r25 + r24,
        ]
    )
    return dy


# ── reference solution ────────────────────────────────────────────────────────
u0 = np.zeros(20)
u0[1] = 0.2
u0[3] = 0.04
u0[6] = 0.1
u0[7] = 0.3
u0[8] = 0.01
u0[16] = 0.007
u0 = jnp.array(u0)

_sol = diffrax.diffeqsolve(
    ODETerm(pollu),
    Kvaerno3(),
    t0=tspan[0],
    t1=tspan[1],
    dt0=1e-3,
    y0=u0,
    saveat=SaveAt(ts=tsteps),
    stepsize_controller=PIDController(rtol=1e-12, atol=1e-6),
    max_steps=100_000,
)
normdata = _sol.ys.T  # shape (20, ntotal)  — matches Julia layout

i_slow = jnp.arange(20)
nslow = 20
yscale = normdata.max(axis=1) - normdata.min(axis=1)  # (20,)
yscale = jnp.where(yscale == 0, 1.0, yscale)

# ── Neural ODE (equinox) ──────────────────────────────────────────────────────
node = 10


class NeuralODE(eqx.Module):
    layers: list

    def __init__(self, key):
        keys = jax.random.split(key, 5)
        act = jax.nn.gelu
        self.layers = [
            eqx.nn.Linear(nslow, node, key=keys[0]),
            eqx.nn.Linear(node, node, key=keys[1]),
            eqx.nn.Linear(node, node, key=keys[2]),
            eqx.nn.Linear(node, nslow, key=keys[3]),
        ]

    def __call__(self, x):
        x = self.layers[0](x)
        x = jax.nn.gelu(x)
        x = self.layers[1](x)
        x = jax.nn.gelu(x)
        x = self.layers[2](x)
        x = jax.nn.gelu(x)
        x = self.layers[3](x)
        return x


key = jax.random.PRNGKey(0)
model = NeuralODE(key)


# ── ODE right-hand side driven by the network ─────────────────────────────────
def dudt(t, u, model):
    return model(u) * yscale / t_end


# ── predict: integrate from 0 → tsteps[sample-1] and save at tsteps[0:sample] ─
def predict_n_ode(model, sample):
    ts = tsteps[:sample]
    term = ODETerm(dudt)
    sol = diffrax.diffeqsolve(
        term,
        Kvaerno3(),
        t0=tsteps[0],
        t1=tsteps[sample - 1],
        dt0=1e-3,
        y0=u0,
        args=model,
        saveat=SaveAt(ts=ts),
        stepsize_controller=PIDController(rtol=1e-6, atol=lb),
        max_steps=200_000,
    )
    return sol.ys.T  # (nslow, sample)


# ── loss: MAE on normalised values ───────────────────────────────────────────
def loss_n_ode(model, sample):
    pred = predict_n_ode(model, sample)
    n = pred.shape[1]
    target = normdata[:, :n]
    return jnp.mean(jnp.abs(pred / yscale[:, None] - target / yscale[:, None]))


# ── optimiser: ADAMW ─────────────────────────────────────────────────────────
opt = optax.adamw(learning_rate=0.005, b1=0.9, b2=0.999, weight_decay=1e-6)
opt_state = opt.init(eqx.filter(model, eqx.is_array))


# ── training step ─────────────────────────────────────────────────────────────
@eqx.filter_jit
def train_step(model, opt_state, sample):
    loss, grads = eqx.filter_value_and_grad(loss_n_ode)(model, sample)

    # match Julia: grad = grad ./ norm(grad) .* grad_max
    arr_grads = eqx.filter(grads, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(arr_grads)
    g_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
    scale = grad_max / (g_norm + 1e-8)
    scaled_grads = jax.tree_util.tree_map(lambda g: g * scale, arr_grads)

    updates, opt_state_new = opt.update(
        scaled_grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    model_new = eqx.apply_updates(model, updates)
    return model_new, opt_state_new, loss, g_norm


# ── checkpoint helpers ────────────────────────────────────────────────────────
def save_ckpt(model, opt_state, list_loss, list_grad, iter_):
    with open("checkpoints/mymodel.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "opt_state": opt_state,
                "list_loss": list_loss,
                "list_grad": list_grad,
                "iter": iter_,
            },
            f,
        )


def load_ckpt():
    with open("checkpoints/mymodel.pkl", "rb") as f:
        d = pickle.load(f)
    return d["model"], d["opt_state"], d["list_loss"], d["list_grad"], d["iter"]


# ── training loop ─────────────────────────────────────────────────────────────
list_loss = []
list_grad = []
iter_ = 1
rng = np.random.default_rng(42)

if is_restart:
    model, opt_state, list_loss, list_grad, iter_ = load_ckpt()
    iter_ += 1

for epoch in range(iter_, n_epoch + 1):
    sample = int(rng.integers(batch_size, ntotal + 1))

    model, opt_state, loss, g_norm = train_step(model, opt_state, sample)
    loss = float(loss)
    g_norm = float(g_norm)

    list_loss.append(loss)
    list_grad.append(g_norm)

    # progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch:6d}  Loss: {loss:.4e}  grad: {g_norm:.2e}")

    # ── plots & checkpoint every n_plot epochs ────────────────────────────────
    if epoch % n_plot == 0:
        pred_np = np.array(predict_n_ode(model, ntotal))
        ts_np = np.array(tsteps)
        nd_np = np.array(normdata)

        # -- species predictions
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        for i, ax in enumerate(axes.flat):
            ax.scatter(ts_np, nd_np[i], s=10, label="data")
            ax.plot(ts_np, pred_np[i], lw=2, label="pred")
            ax.set_title(f"y{i + 1}", fontsize=8)
            ax.set_box_aspect(1)
            if i == 0:
                ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig("plots/pred.png", dpi=100)
        plt.close(fig)

        # -- loss & grad norm
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.semilogy(list_loss)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss")
        ax2.semilogy(list_grad)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Gradient Norm")
        ax2.set_title("Grad Norm")
        fig.tight_layout()
        fig.savefig("plots/loss_grad.png", dpi=100)
        plt.close(fig)

        save_ckpt(model, opt_state, list_loss, list_grad, epoch)
