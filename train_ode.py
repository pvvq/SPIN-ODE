#!/usr/bin/env python
# coding: utf-8

# Train a learnable ODE with solver on concentration
# usage: see `python train_ode.py --help`

from typing import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import matplotlib.pyplot as plt

import train_utils
import chem_data as cd
import network as nt
import plots.plot as pp


config = train_utils.load_config()
print("config:", config)
np.printoptions(precision=0, linewidth=300)
torch.set_default_dtype(torch.float64)
torch.set_printoptions(profile='full', precision=4, linewidth=1000)
jax.config.update("jax_enable_x64", True)
print(jax.devices(), jax.default_backend())

# Data =========================================================================

if config['chem'] == 'rober':   chem = cd.ROBER()
elif config['chem'] == 'pollu': chem = cd.POLLU()
elif config['chem'] == 'toy':   chem = cd.TOY()

y_arr, t_arr = chem.data(config['n_series'], rand=config['rand_init'])
if 'sample' in config:  # resample in time dim, reduce by a factor of 'step'
    y_arr = y_arr[:,::config['sample']['step'],:]
    t_arr = t_arr[:,::config['sample']['step']]

scale = {
    'yMax' : np.max(y_arr, axis=(0,1)),
    'yMin' : np.min(y_arr, axis=(0,1)),
    'tScale' : t_arr[0,-1] - t_arr[0,0],  # same tscale for every traj
}
scale['yScale'] = np.where(scale['yMax']-scale['yMin'] == 0.0, scale['yMax'], scale['yMax']-scale['yMin'])
scale['ytScale'] = scale['yScale'] / scale['tScale']
finit_dy = np.gradient(y_arr, t_arr[0], axis=1)
scale['dyMax'] = np.max(finit_dy, axis=(0,1))
scale['dyMin'] = np.min(finit_dy, axis=(0,1))
scale['dyScale'] = np.where(scale['dyMax']-scale['dyMin'] == 0.0, scale['dyMax'], scale['dyMax']-scale['dyMin'])
finit_ddy = np.gradient(finit_dy, t_arr[0], axis=1)
scale['ddyMax'] = np.max(finit_ddy, axis=(0,1))
scale['ddyMin'] = np.min(finit_ddy, axis=(0,1))
scale['ddyScale'] = np.where(scale['ddyMax']-scale['ddyMin'] == 0.0, scale['ddyMax'], scale['ddyMax']-scale['ddyMin'])
# fig = plot_series(y_arr[0])
# fig.savefig("finit_y.png", dpi=300)
# fig = plot_series(finit_dy[0])
# fig.savefig("finit_dy.png", dpi=300)
# fig = plot_series(finit_ddy[0])
# fig.savefig("finit_ddy.png", dpi=300)
# plt.close(fig)

# coll_dataset = CollocateDataset(y_arr, t_arr)
yt_dataset = cd.ChuckDataset(
    y_arr, t_arr,
    chuck_len=config['chuck']['chuck_len'],
    stride_len=config['chuck']['stride_len'],
    ratio=config['chuck']['ratio']
)
yt_val_dataset = cd.ChuckDataset(y_arr[-1:], t_arr[-1:])

# coll_dataloader = DataLoader(coll_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=jax_collate)
yt_dataloader = DataLoader(yt_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=cd.jax_collate)
yt_val_dataloader = DataLoader(yt_val_dataset, batch_size=1, shuffle=False, collate_fn=cd.jax_collate)
last_traj = {'time': jnp.asarray(t_arr[-1]), 'conc': jnp.asarray(y_arr[-1])}

inc_dataloaders = [
    DataLoader(cd.ChuckDataset(y_arr[:,:trunc_len,:], t_arr[:,:trunc_len]),
               batch_size=config['batch_size'], shuffle=True, collate_fn=cd.jax_collate)
    for trunc_len in range(20, y_arr.shape[1], 20)
]

for k,v in next(iter(yt_dataloader)).items():
    print(f"batch['{k}': shape{v.shape}]")

yt_dataloader.dataset.rand_sample()

# Model ========================================================================

# Instantiate the ODE.
if config['ODE'] == "ScaleMLP":
    ode = nt.ScaleMLP(
        chem.num_spc, scale,
        hidden_size=128, rngs=nnx.Rngs(0),
    )
elif config['ODE'] == "LogRateLaw":
    ode = nt.LogRateLaw(
        chem.stoi_reac, chem.stoi_prod,
        chem.RO2_IDX, chem.RO2_K_IDX,
        k=chem.rconst
    )

if config['restore']:
    ode = train_utils.restore_ckpt(config, ode)

if config['chem'] == "pollu" and config['restore'] is not None:
    # OH cycling rate coefficients are provided as they are well studied to help network converge
    pollu_k_OH = np.exp(np.random.normal(size=chem.num_react))
    pollu_k_OH[14:19] = [
        .480e+7,  # 15. O3P -> O3
        .350e-3,  # 16. O3 -> O1D
        .175e-1,  # 17. O3 -> O3P
        .100e+9,  # 18. O1D -> 2OH
        .444e12,  # 19. O1D -> O3P
    ]
    guess_k = pollu_k_OH
    print(ode.get_k(), nt.LogMAELoss(ode.get_k(), jnp.asarray(chem.rconst)))
    print("init OH", guess_k[14:19])
    ode.ln_k.value = ode.ln_k.value.at[14:19].set(jnp.log(jnp.asarray(guess_k[14:19]).reshape(-1,1)))
    print(ode.get_k(), nt.LogMAELoss(ode.get_k(), jnp.asarray(chem.rconst)))

true_rate_ode = nt.LogRateLaw(
    chem.stoi_reac, chem.stoi_prod,
    chem.RO2_IDX, chem.RO2_K_IDX,
    k=chem.rconst
)

optimizer = nnx.Optimizer(
    ode,
    optax.chain(
        optax.adam(config['learning_rate']),
        optax.contrib.reduce_on_plateau(patience=config['n_epochs']*0.1*len(yt_dataloader))
    ),
    wrt=nnx.Param,
)

# Training =====================================================================

solver = nt.ode_solver()

def loss_fn(ode: Callable, conc: jax.Array, t: jax.Array) -> jax.Array:
    pred_y = solver(ode, t, conc[0,:])
    
    loss_func = nt.ScaleMSELoss
    loss = loss_func(pred_y[1:,:], conc[1:,:], jnp.asarray(scale['yScale']))
    grad_1st = jnp.gradient(pred_y, t, axis=0)
    grad_2nd = jnp.gradient(grad_1st, t, axis=0)
    if config['phy_loss'] and 'grad_sim' in config['phy_loss']:
        loss_grad1st = loss_func(
            grad_1st,
            jnp.gradient(conc, t, axis=0),
            jnp.asarray(scale['dyScale'])
        )
        loss_grad2nd = loss_func(
            grad_2nd,
            jnp.gradient(jnp.gradient(conc), t, axis=0),
            jnp.asarray(scale['ddyScale'])
        )
        alpha = config['phy_loss']['grad_sim']['alpha']
        loss = loss + loss_grad1st * alpha + loss_grad2nd * alpha
    if config['phy_loss'] and 'tv' in config['phy_loss']:
        alpha = config['phy_loss']['tv']['alpha']
        window_size = config['phy_loss']['tv']['window_size']
        loss += nt.TVLoss1D(grad_1st, jnp.asarray(scale['dyScale']), window_size) * alpha
        loss += nt.TVLoss1D(grad_2nd, jnp.asarray(scale['ddyScale']), window_size) * alpha
    return loss

def batch_loss_fn(
        ode: Callable,
        batch: dict[str, jax.Array],
    ):
    batch_loss = nnx.vmap(
        loss_fn, in_axes=(None, 0, 0), out_axes=0,
    )(ode, batch['conc'], batch['time'])
    return jnp.mean(batch_loss)

@nnx.jit
def train_step(ode, optimizer, batch):
    grad_fn = nnx.value_and_grad(batch_loss_fn)
    loss, grads = grad_fn(ode, batch)
    optimizer.update(grads, value=loss)
    return loss

async_worker = ThreadPoolExecutor(max_workers=8)

def eval_k(ode, batch, step, logger):
    pred_k = ode.get_k()
    true_k = jnp.asarray(chem.rconst)
    err_k = nt.LogMAELoss(pred_k, true_k)
    
    pred_k, err_k = np.asarray(pred_k), np.asarray(err_k)
    def _plot():
        fig = pp.plot_k([pred_k, chem.rconst],["pred_k", "true_k"], chem.num_react)
        logger.add_figure('pred_true_k', fig, step)
        logger.add_text('pred_k', np.array2string(pred_k), step)
        logger.add_scalar('val_err_k', err_k, step)
        plt.close(fig)
    async_worker.submit(_plot)

def eval_y(ode, batch, step, logger):
    sample = last_traj
    jitted_solver = jax.jit(solver, static_argnames="ode")
    pred_y = jitted_solver(ode, sample['time'], sample['conc'][0])
    err_y = nt.ScaleMSELoss(pred_y, sample['conc'], scale['yScale'])
    
    pred_y, true_y, err_y = np.asarray(pred_y), np.asarray(sample['conc']), np.asarray(err_y)
    def _plot():
        fig = pp.plot_series(y=pred_y, yy=true_y)
        logger.add_figure('pred_true_y', fig, step)
        logger.add_scalar('val_err_y', err_y, step)
        plt.close(fig)
    async_worker.submit(_plot)

def eval_dy(ode, batch, step, logger):
    sample = last_traj
    def batch_ode(ode):
        return nnx.jit(nnx.vmap(
            lambda t, c: ode(t, c),
            in_axes=(0, 0), out_axes=0,
        ))
    ode_dcdt = batch_ode(ode)(sample['time'], sample['conc'])
    true_dcdt = batch_ode(true_rate_ode)(sample['time'], sample['conc'])
    err_dy = nt.ScaleMSELoss(ode_dcdt, true_dcdt, scale['dyScale'])

    ode_dcdt, true_dcdt, err_dy = np.asarray(ode_dcdt), np.asarray(true_dcdt), np.asarray(err_dy)
    def _plot():
        fig = pp.plot_series(y=ode_dcdt, yy=true_dcdt)
        logger.add_figure('pred_true_dy', fig, step)
        logger.add_scalar('val_err_dy', err_dy, step)
        plt.close(fig)
    async_worker.submit(_plot)

def val_step(ode, val_dataloader, step, logger):
    if isinstance(ode, nt.LogRateLaw) or isinstance(ode, nt.PowerRateLaw):
        eval_k(ode, None, step, logger)
    eval_y(ode, None, step, logger)
    eval_dy(ode, None, step, logger)

logger, checkpointer = train_utils.build_logging(config)
train_utils.train_loop(
    ode, optimizer,
    train_dataloader=yt_dataloader, val_dataloader=yt_val_dataloader,
    train_step=train_step, val_step=val_step,
    logger=logger, checkpointer=checkpointer,
    config=config,
)