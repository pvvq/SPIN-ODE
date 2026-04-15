import sys
import yaml
import argparse
from pathlib import Path
from shutil import copyfile

import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import preservation_policy as prsv_plcy
from orbax.checkpoint.checkpoint_managers import save_decision_policy as save_plcy


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="path to YAML config")
    parser.add_argument("--target", "-t", type=str, default="default", help="target in YAML config")
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True, help="enable training")
    parser.add_argument("--infer", action=argparse.BooleanOptionalAction, default=True, help="enable inference")
    parser.add_argument("--save_dir", "-s", type=str, help="dir to save logs and checkpoints (erase if exist)")
    parser.add_argument("--redirect", "-r", type=str, help="file to redirect stdout")
    args = parser.parse_args()
    return args

def load_config():
    args = parse_cmd_args()

    # Load YAML
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f).get(args.target)

    if config_dict is None:
        raise ValueError(f"Target '{args.target}' not found in config.")

    config_dict["train"] = args.train
    config_dict["infer"] = args.infer

    if args.save_dir:
        save_dir = Path(args.save_dir).absolute()
        ocp.test_utils.erase_and_create_empty(save_dir)
        copyfile(args.config, save_dir / Path(args.config).name)
        config_dict["save_dir"] = save_dir
    else:
        config_dict["save_dir"] = None

    if args.redirect:
        sys.stdout = open(args.redirect, "w", buffering=1)

    return config_dict

def get_checkpoint_manager(ckpt_dir, interval=1, keep=1, get_metric_fn=lambda m: m):
    # remember to call ckpt_mngr.wait_until_finished(); ckpt_mngr.close()
    return ocp.CheckpointManager(
        ckpt_dir,
        options=ocp.CheckpointManagerOptions(
            save_decision_policy=save_plcy.FixedIntervalPolicy(interval),
            preservation_policy=prsv_plcy.AnyPreservationPolicy(
                policies=[
                    prsv_plcy.LatestN(n=keep),
                    prsv_plcy.BestN(get_metric_fn=get_metric_fn, reverse=False, n=1),
                ]),
        ),
    )

# use new args API
def standard_save(ckpt_mngr, step, state, metrics):
    ckpt_mngr.save(step, args=ocp.args.StandardSave(state), metrics=metrics)

def standard_restore(ckpt_mngr, step, abstract_state):
    # ckpt_mngr.all_steps(); ckpt_mngr.latest_step(); ckpt_mngr.best_step()
    return ckpt_mngr.restore(step, args=ocp.args.StandardRestore(abstract_state))
