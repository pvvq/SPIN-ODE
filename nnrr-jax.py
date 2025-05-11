#!/usr/bin/env python
# coding: utf-8

# usage: see `python nnrr-jax.py --help`

# # Neural Reaction Network


#%%
from utils import *
config = load_config()
print("config:", config)

from datetime import datetime
from pathlib import Path
from shutil import copyfile

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp

from chem_data import TOY, ROBER, POLLU, AEDataset, CollocateDataset, ChuckDataset
from nn_jax import *

np.printoptions(precision=0, linewidth=300)
torch.set_default_dtype(torch.float64)
torch.set_printoptions(profile='full', precision=4, linewidth=1000)
jax.config.update("jax_enable_x64", True)
print(jax.devices(), jax.default_backend())

# ## Data

if config['chem'] == 'rober':   chem = ROBER()
elif config['chem'] == 'pollu': chem = POLLU()
elif config['chem'] == 'toy':   chem = TOY(data_dir="results_t100_dt1_10")

y_arr, t_arr = chem.data(config['n_series'], config['rand_init'])
if 'sample' in config:  # resample in time dim, reduce by a factor of 'step'
    y_arr = y_arr[:,::config['sample']['step'],:]
    t_arr = t_arr[:,::config['sample']['step']]
# y_arr = y_arr + 1e-30   # prevent 0 for log

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
fig = plot_series(y_arr[0])
fig.savefig("finit_y.png", dpi=300)
fig = plot_series(finit_dy[0])
fig.savefig("finit_dy.png", dpi=300)
fig = plot_series(finit_ddy[0])
fig.savefig("finit_ddy.png", dpi=300)
plt.close(fig)

# coll_dataset = CollocateDataset(y_arr, t_arr)
yt_dataset = ChuckDataset(
    y_arr, t_arr,
    chuck_len=config['chuck']['chuck_len'],
    stride_len=config['chuck']['stride_len'],
    ratio=config['chuck']['ratio']
)
yt_val_dataset = ChuckDataset(y_arr[-1:], t_arr[-1:])

# coll_dataloader = DataLoader(coll_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=jax_collate)
yt_dataloader = DataLoader(yt_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=jax_collate)
yt_val_dataloader = DataLoader(yt_val_dataset, batch_size=1, shuffle=False, collate_fn=jax_collate)
last_traj = {'time': jnp.array(t_arr[-1]), 'conc': jnp.array(y_arr[-1])}

inc_dataloaders = [
    DataLoader(ChuckDataset(y_arr[:,:trunc_len,:], t_arr[:,:trunc_len]),
               batch_size=config['batch_size'], shuffle=True, collate_fn=jax_collate)
    for trunc_len in range(20, y_arr.shape[1], 20)
]

for k,v in next(iter(yt_dataloader)).items():
    print(f"batch['{k}': shape{v.shape}]")


# ## Training

def loss_fn(model, batch):
    init_conc = batch['conc'][:,0,:]
    t = batch['time'][0]  # use shared time stamp
    pred_y = model(init_conc, t)
    loss_func = ScaleMSELoss
    loss = loss_func(pred_y[:,1:,:], batch['conc'][:,1:,:], jnp.array(scale['yScale']))
    grad_1st = gradient(pred_y)
    grad_2nd = gradient(grad_1st)
    if config['phy_loss'] and 'grad_sim' in config['phy_loss']:
        loss_grad1st = loss_func(
            grad_1st,
            gradient(batch['conc']),
            jnp.array(scale['dyScale'])
        )
        loss_grad2nd = loss_func(
            grad_2nd,
            gradient(gradient(batch['conc'])),
            jnp.array(scale['ddyScale'])
        )
        alpha = config['phy_loss']['grad_sim']['alpha']
        loss = loss + loss_grad1st * alpha + loss_grad2nd * alpha
    if config['phy_loss'] and 'tv' in config['phy_loss']:
        alpha = config['phy_loss']['tv']['alpha']
        window_size = config['phy_loss']['tv']['window_size']
        loss += TVLoss1D(grad_1st, jnp.array(scale['dyScale']), window_size) * alpha
        loss += TVLoss1D(grad_2nd, jnp.array(scale['ddyScale']), window_size) * alpha
    return loss

def loss_fn_coll(model, batch):
    conc = batch['conc']
    t = batch['time']
    pred_dcdt = model(t, conc)
    estm_dcdt = batch['dcdt']
    return SMSPELoss(pred_dcdt, estm_dcdt)

@nnx.jit  # automatic state management for JAX transforms
def train_step(model, optimizer, batch):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    optimizer.update(grads, value=loss)  # in-place updates
    return loss

def eval_k(model, batch, step):
    pred_k = model.get_k()
    true_k = jnp.array(chem.rconst)
    err_k = LogMAELoss(pred_k, true_k)
    fig = plot_k([jax.device_get(pred_k), chem.rconst],["pred_k", "true_k"], chem.num_react)
    logger.add_figure('pred_true_k', fig, step)
    logger.add_text('pred_k', jnp.array_str(pred_k), step)
    plt.close(fig)
    return err_k

def eval_y(model, batch, step):
    pred_y = model(batch['conc'][:,0,:], batch['time'][0])
    err_y = ScaleMSELoss(pred_y, batch['conc'], scale['yScale'])
    fig = plot_series(pred_y.squeeze(), batch['conc'].squeeze())
    logger.add_figure('pred_true_y', fig, step)
    plt.close(fig)
    return err_y

def eval_dy(ode, crnn_true, traj, step):
    ode_dcdt = ode(traj['time'], traj['conc'])
    true_dcdt = crnn_true(traj['time'], traj['conc'])
    err_dy = ScaleMSELoss(ode_dcdt, true_dcdt, scale['dyScale'])
    fig = plot_series(ode_dcdt, true_dcdt)
    logger.add_figure('pred_true_dy', fig, step)
    plt.close(fig)
    return err_dy

# ## Logging
if config['log']:
    log_dir = Path(f"./").absolute() / config['base_dir'] / config['name'] \
        / (datetime.now().strftime('%Y%m%d-%H%M%S') + "_" + config['version'])
    log_dir.mkdir(parents=True, exist_ok=True)
    copyfile(config['config'], log_dir / "config.yaml")
    logger = SummaryWriter(log_dir)
    logger.add_text("config", yaml.dump(config, sort_keys=False))
    checkpointer = ocp.CheckpointManager(
        log_dir / 'checkpoints',
        ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=config['ckpt_keep'],
            save_interval_steps=config['ckpt_interval'],
        )
    )
else:
    log_dir = Path("./").absolute()
    logger = DummyLogger()
    checkpointer = DummyCheckpointer()

if config['restore']:
    # restore from checkpoint
    restore_checkpointer = ocp.CheckpointManager(
        Path(config['restore']).absolute() / 'checkpoints',
        ocp.PyTreeCheckpointer(),
    )
    if config['nODE'] == "ScaleMLP":
        dummy_ode = ScaleMLP(
            chem.num_spc, chem.num_react, scale,
            hidden_size=128, rngs=nnx.Rngs(0),
        )
    elif config['nODE'] == "CRNN":
        dummy_ode = CRNN(
            chem.num_spc, chem.num_react,
            chem.coef_in, chem.coef_out,
            chem.RO2_IDX, chem.RO2_K_IDX,
            # k=guess_k
        )
    abstract_model = nnx.eval_shape(lambda: 
        dummy_ode
    )
    graphdef, abstract_state = nnx.split(abstract_model)
    target_ckpt = {
        'ode': abstract_state,
        'step': 0,
    }
    restored_ckpt = restore_checkpointer.restore(
        restore_checkpointer.latest_step(),
        items=target_ckpt
    )
    ode = nnx.merge(graphdef, restored_ckpt['ode'])
    model = NeuralODE(ode)
else:
    # Instantiate the model.
    if config['nODE'] == "ScaleMLP":
        ode = ScaleMLP(
            chem.num_spc, chem.num_react, scale,
            hidden_size=128, rngs=nnx.Rngs(0),
        )
    elif config['nODE'] == "CRNN":
        ode = CRNN(
            chem.num_spc, chem.num_react,
            chem.coef_in, chem.coef_out,
            chem.RO2_IDX, chem.RO2_K_IDX,
        )
    model = NeuralODE(ode)

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
    print(model.ode.get_k(), LogMAELoss(model.ode.get_k(), jnp.array(chem.rconst)))
    print("init OH", guess_k[14:19])
    model.ode.ln_k.value = model.ode.ln_k.value.at[14:19].set(jnp.log(jnp.array(guess_k[14:19]).reshape(-1,1)))
    print(model.ode.get_k(), LogMAELoss(model.ode.get_k(), jnp.array(chem.rconst)))

crnn_true = CRNN(
    chem.num_spc, chem.num_react,
    chem.coef_in, chem.coef_out,
    chem.RO2_IDX, chem.RO2_K_IDX,
    k=chem.rconst
)

optimizer = nnx.Optimizer(
    model,
    optax.chain(
        optax.adam(config['learning_rate']),
        optax.contrib.reduce_on_plateau(patience=config['n_epochs']*0.1*len(yt_dataloader))
    )
)

#%%
err_k = None
n_step = 0
n_epoch = 0
bar = tqdm(range(0, config['n_epochs']), desc=f"Epoch", initial=0)
for i_epoch, epoch in enumerate(bar):
    # train
    model.train()
    mean_loss = 0
    yt_dataloader.dataset.rand_sample()
    for batch in yt_dataloader:

        loss = train_step(model, optimizer, batch)
        mean_loss += loss

        n_step += 1
    mean_loss /= len(yt_dataloader)
    logger.add_scalar("y_loss_epoch", np.asarray(mean_loss), n_epoch)
    lr_scale = optax.tree_utils.tree_get(optimizer.opt_state, "scale")
    logger.add_scalar("lr_scale", np.asarray(lr_scale*config['learning_rate']), n_epoch)
    logger.add_scalar("epochs", np.asarray(n_epoch), n_epoch)

    # evaluation
    model.eval()
    if (epoch + 1) % config['val_interval'] == 0:
        if isinstance(model.ode, CRNN):
            err_k = eval_k(model, None, epoch)
            logger.add_scalar('val_err_k', np.asarray(err_k), n_epoch)
        err_y = eval_y(model, next(iter(yt_val_dataloader)), n_epoch)
        logger.add_scalar('val_err_y', np.asarray(err_y), n_epoch)
        err_dy = eval_dy(model.ode, crnn_true, last_traj, n_epoch)
        logger.add_scalar('val_err_dy', np.asarray(err_dy), n_epoch)

    # checkpointing
    ckpt = {
        'model': nnx.split(model)[1],
        'step': epoch,
    }
    checkpointer.save(epoch+1, ckpt)

    postfix = {
        'loss':f"{np.asarray(mean_loss):.4e}",
        'err_k':f"{np.asarray(err_k):.4e}" if err_k is not None else None,
    }
    bar.set_postfix(postfix)
    n_epoch += 1
# %%
