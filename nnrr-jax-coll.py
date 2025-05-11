#!/usr/bin/env python
# coding: utf-8

# usage: see `python nnrr-jax.py --help`

# # Neural Reaction Network

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
from scipy.signal import savgol_filter

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

y_arr, t_arr = chem.data(config['n_series'])  # np: [n, t, s], [n, t]
# omit beginning induction period
# tunc_idx = 2
# y_arr = y_arr[:,tunc_idx:-tunc_idx,:]
# t_arr = t_arr[:,tunc_idx:-tunc_idx]
# y_arr = y_arr + 1e-30   # prevent 0 for log

scale = {
    'yMax' : np.max(y_arr, axis=(0,1)),
    'yMin' : np.min(y_arr, axis=(0,1)),
    'tScale' : t_arr[0,-1] - t_arr[0,0],  # same tscale for every traj
}
scale['yScale'] = np.where(scale['yMax']-scale['yMin'] == 0.0, scale['yMax'], scale['yMax']-scale['yMin'])
scale['ytScale'] = scale['yScale'] / scale['tScale']

# coll_dataset = CollocateDataset(y_arr[:-1], t_arr[:-1])
yt_dataset = ChuckDataset(
    y_arr, t_arr,
    chuck_len=config['chuck']['chuck_len'],
    stride_len=config['chuck']['chuck_len'],
    ratio=config['chuck']['ratio']
)
# yt_val_dataset = ChuckDataset(y_arr[-1:], t_arr[-1:], chuck_len=y_arr.shape[1])

# coll_dataloader = DataLoader(coll_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=jax_collate)
# yt_dataloader = DataLoader(yt_dataset, batch_size=config['batch_size'], shuffle=True, c/ollate_fn=jax_collate)
last_traj = {'time': jnp.array(t_arr[-1]), 'conc': jnp.array(y_arr[-1])}
yt_val_dataloader = DataLoader(yt_dataset, batch_size=1, shuffle=False, collate_fn=jax_collate)

if config['chem'] == 'pollu':
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
else:
    guess_k = None

# ## Training

def loss_fn(model, batch):
    init_conc = batch['conc'][:,0,:]
    t = batch['time'][0]  # use shared time stamp
    pred_y = model(init_conc, t)
    return ScaleMSELoss(pred_y[:,1:,:], batch['conc'][:,1:,:], jnp.array(scale['yScale']))
    # return SMSPELoss(pred_y[:,1:,:], batch['conc'][:,1:,:])

def loss_fn_coll(model, batch):
    conc = batch['conc']
    t = batch['time']
    pred_dcdt = model(t, conc)
    estm_dcdt = batch['dcdt']
    return jnp.log(ScaleMSELoss(pred_dcdt, estm_dcdt, jnp.array(scale['dyScale'])))
    # return SMSPELoss(pred_dcdt, estm_dcdt)

@nnx.jit  # automatic state management for JAX transforms
def train_step(model, optimizer, batch):
    grad_fn = nnx.value_and_grad(loss_fn_coll)
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
        NeuralODE(dummy_ode)
    )
    graphdef, abstract_state = nnx.split(abstract_model)
    target_ckpt = {
        'model': abstract_state,
        'step': 0,
    }
    restored_ckpt = restore_checkpointer.restore(
        restore_checkpointer.latest_step(),
        items=target_ckpt
    )
    model = nnx.merge(graphdef, restored_ckpt['model'])
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

model.crnn_true = CRNN(
    chem.num_spc, chem.num_react,
    chem.coef_in, chem.coef_out,
    chem.RO2_IDX, chem.RO2_K_IDX,
    k = chem.rconst
)
model.crnn = CRNN(
    chem.num_spc, chem.num_react,
    chem.coef_in, chem.coef_out,
    chem.RO2_IDX, chem.RO2_K_IDX,
    k = guess_k
)

mlp_dcdt = model.ode(last_traj['time'], last_traj['conc'])
true_dcdt = model.crnn_true(last_traj['time'], last_traj['conc'])
fig = plot_series(mlp_dcdt, true_dcdt)
fig.savefig("fit_dydt.png", dpi=300)
plt.close(fig)
pred_y = model(
    last_traj['conc'][None, 0], 
    last_traj['time'],
)
print(mlp_dcdt.shape, true_dcdt.shape, pred_y.shape)
fig = plot_series(pred_y.squeeze(), last_traj['conc'])
fig.savefig("fit_y.png", dpi=300)

if 'diff_interpolate' in config['dydt']:  # opt 1: diff from interpolated fitted trajectory
    interpolate_ratio = config['dydt']['diff_interpolate']['ratio']
    yy_list = []  # interpolated concentration trajectory
    tt_list = []  # interpolated time stamp
    for chunk in yt_dataset:
        time, conc = chunk['time'], chunk['conc']
        if config['chem'] == "rober":  # log space time stamps
            tt = np.logspace(np.log10(time[0]), np.log10(time[-1]), num=time.shape[-1] * interpolate_ratio)
        else:  # linear space time stamps
            tt = np.linspace(time[0], time[-1], num=time.shape[-1] * interpolate_ratio)
        yy = model(conc[None, 0], tt).squeeze()
        tt_list.append(tt)
        yy_list.append(np.array(yy))
    yy_arr, tt_arr = np.array(yy_list), np.array(tt_list)
    # tt_arr = np.concatenate(tt_list)[None]  # full trajectory time
    # yy_arr = np.concatenate(yy_list)[None]  # full trajectory concentration
    # yy_arr = savgol_filter(yy_arr, window_length=31, polyorder=3, axis=1)
    print(yy_arr.shape, tt_arr.shape)
    coll_dataset = CollocateDataset(yy_arr, tt_arr)
elif 'diff_origin' in config['dydt']:  # opt 2: diff from fitted trajectory
    coll_dataset = CollocateDataset(y_arr, t_arr)
elif 'mlp_output' in config['dydt']:  # opt 3: from learned time deriv
    coll_dataset = CollocateDataset(y_arr, t_arr, np.asarray(mlp_dcdt[None,:,:]))

coll_dataloader = DataLoader(coll_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=jax_collate)

scale['dyMax'] = np.max(coll_dataset.dy_arr.numpy(), axis=(0,1))
scale['dyMin'] = np.min(coll_dataset.dy_arr.numpy(), axis=(0,1))
scale['dyScale'] = np.where(scale['dyMax']-scale['dyMin'] == 0.0, scale['dyMax'], scale['dyMax']-scale['dyMin'])

for i, (y, dy, t) in enumerate(zip(coll_dataset.y_arr, coll_dataset.dy_arr, coll_dataset.t_arr)):
    if i == 0:
        fig1 = plot_series(y.numpy(), y_arr[-1], t=t.numpy(), t2=t_arr[-1])
        fig2 = plot_series(dy.numpy(), true_dcdt, t=t.numpy(), t2=t_arr[-1])
    else:
        fig1 = plot_series(y.numpy(), t=t.numpy(), fig=fig1)
        fig2 = plot_series(dy.numpy(), t=t.numpy(), fig=fig2)
fig1.savefig("coll_y.png", dpi=300)
fig2.savefig("coll_dy.png", dpi=300)
plt.close(fig1)
plt.close(fig2)


for k,v in next(iter(coll_dataloader)).items():
    print(f"batch['{k}': shape {v.shape}]")


optimizer = nnx.Optimizer(
    model.crnn,
    optax.chain(
        optax.adam(config['learning_rate']),
        optax.contrib.reduce_on_plateau(patience=config['n_epochs']*0.1*len(coll_dataloader))
    )
)

model.ode = model.crnn  # reset ode to crnn for validation on y

err_k = None
n_step = 0
bar = tqdm(range(0, config['n_epochs']), desc=f"Epoch", initial=0)
for i_epoch, epoch in enumerate(bar):
    if (epoch + 1) % config['val_interval'] == 0:
        err_k = eval_k(model.crnn, None, epoch)
        logger.add_scalar('val_err_k', np.asarray(err_k), i_epoch)
        err_y = eval_y(model, next(iter(yt_val_dataloader)), epoch)
        logger.add_scalar('val_err_y', np.asarray(err_y), i_epoch)
        err_dy = eval_dy(model.ode, model.crnn_true, last_traj, i_epoch)
        logger.add_scalar('val_err_dy', np.asarray(err_dy), i_epoch)

    mean_loss = 0
    for batch in coll_dataloader:

        loss = train_step(model.crnn, optimizer, batch)
        mean_loss += loss

        n_step += 1
    mean_loss /= len(coll_dataloader)
    logger.add_scalar('y_loss_epoch', np.asarray(loss), i_epoch)
    lr_scale = optax.tree_utils.tree_get(optimizer.opt_state, "scale")
    logger.add_scalar("lr_scale", np.asarray(lr_scale*config['learning_rate']), i_epoch)
    logger.add_scalar("epochs", np.asarray(i_epoch), i_epoch)

    # checkpointing
    ckpt = {
        'ode': nnx.split(model.ode)[1],
        'step': epoch,
    }
    checkpointer.save(epoch+1, ckpt)

    postfix = {
        'loss':f"{np.asarray(mean_loss):.4e}",
        'err_k':f"{np.asarray(err_k):.4e}" if err_k is not None else None,
    }
    bar.set_postfix(postfix)
