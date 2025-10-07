import yaml
import argparse
from pathlib import Path
from datetime import datetime
from shutil import copyfile
from typing import Callable

import numpy as np
from flax import nnx
from torch.utils.tensorboard import SummaryWriter
import orbax.checkpoint as ocp
from tqdm import tqdm


class CMDArgsParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--config", "--cf", required=True, type=str, help="Path to YAML config")
        self.add_argument("--target", "-t", required=True, help="target in YAML config")
        self.add_argument('--log', action=argparse.BooleanOptionalAction, default=True, help="log to tensorboard")

def load_config():
    parser = CMDArgsParser()
    args = parser.parse_args()

    # Load YAML
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f).get(args.target)
        
    if config_dict is None:
        raise ValueError(f"Target '{args.target}' not found in config.")

    # Override YAML with non-None CLI args
    for key, value in vars(args).items():
        if value is not None:
            config_dict[key] = value

    return config_dict

class DummyLogger:
    def add_scalar(self, *args, **kwargs): pass
    def add_figure(self, *args, **kwargs): pass
    def add_text(self, *args, **kwargs): pass

class DummyCheckpointer:
    def save(self, *args, **kwargs): pass

def build_logging(config: dict):
    if config['log']:
        log_dir = Path(f"./").absolute() / config['base_dir'] / config['name'] \
            / (datetime.now().strftime('%Y%m%d-%H%M%S') + "_" + config['version'])
        log_dir.mkdir(parents=True, exist_ok=True)
        copyfile(config['config'], log_dir / "config.yaml")
        logger = SummaryWriter(log_dir)
        logger.add_text("config", yaml.dump(config, sort_keys=False))
        checkpointer = ocp.CheckpointManager(
            log_dir / "checkpoints",
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

    return logger, checkpointer

def restore_ckpt(config: dict, empty_model: nnx.Module):
    # restore from checkpoint
    restore_checkpointer = ocp.CheckpointManager(
        Path(config['restore']).absolute() / "checkpoints",
        ocp.PyTreeCheckpointer(),
    )

    abstract_model = nnx.eval_shape(lambda: 
        empty_model
    )
    graphdef, abstract_state = nnx.split(abstract_model)
    target_ckpt = {
        'model': abstract_state,
    }
    restored_ckpt = restore_checkpointer.restore(
        restore_checkpointer.latest_step(),
        items=target_ckpt
    )
    restored_model = nnx.merge(graphdef, restored_ckpt['model'])
    return restored_model

def train_loop(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        train_dataloader,
        val_dataloader,
        train_step: Callable,
        val_step: Callable,
        logger: SummaryWriter | DummyLogger,
        checkpointer: ocp.CheckpointManager | DummyCheckpointer,
        config: dict,
    ):
    n_step = 0
    bar = tqdm(range(0, config['n_epochs']), desc=f"Epochs", initial=0)
    for epoch in bar:
        # === train ===
        model.train()
        mean_loss = 0
        for batch in train_dataloader:

            loss = train_step(model, optimizer, batch)
            mean_loss += loss

            n_step += 1
        mean_loss /= len(train_dataloader)
        logger.add_scalar('loss', np.asarray(mean_loss), epoch)

        # evaluation
        if (epoch) % config['val_interval'] == 0:
            model.eval()
            val_step(model, val_dataloader, epoch, logger)

        # checkpointing
        checkpointer.save(
            epoch,
            {
                'model': nnx.split(model)[1],
            },
        )

        bar.set_postfix({
            'loss': f"{np.asarray(mean_loss):.4e}",
        })