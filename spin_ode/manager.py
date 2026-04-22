import sys
import threading
import yaml
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import preservation_policy as prsv_plcy
from orbax.checkpoint.checkpoint_managers import save_decision_policy as save_plcy
import numpy as np
import matplotlib.pyplot as plt


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="path to YAML config"
    )
    parser.add_argument(
        "--target", "-t", type=str, help="target in YAML config"
    )
    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable training",
    )
    parser.add_argument(
        "--infer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable inference",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="resume from `resume_ckpt` or else `save_dir`",
    )
    parser.add_argument(
        "--save_dir",
        "-s",
        type=str,
        help="dir to save logs and checkpoints",
    )
    parser.add_argument("--redirect", "-r", type=str, help="file to redirect stdout")
    args = parser.parse_args()
    return args


def load_config():
    args = parse_cmd_args()

    # Load YAML
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    if args.target is not None:
        config_dict = config_dict.get(args.target)
        if config_dict is None:
            raise ValueError(f"Target '{args.target}' not found in config.")

    config_dict["train"] = args.train
    config_dict["infer"] = args.infer
    config_dict["resume"] = args.resume

    if args.save_dir:
        save_dir = Path(args.save_dir).absolute()
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
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
                ]
            ),
        ),
    )


# use new args API
def standard_save(ckpt_mngr, step, state: dict, metrics=None):
    # composite save first level `state`
    ckpt_mngr.save(
        step,
        args=ocp.args.Composite(
            **{k: ocp.args.StandardSave(v) for k, v in state.items()}
        ),
        metrics=metrics,
    )


def standard_restore(ckpt_mngr, step, abstract_state: dict):
    # ckpt_mngr.all_steps(); ckpt_mngr.latest_step(); ckpt_mngr.best_step()
    composite = ckpt_mngr.restore(
        step,
        args=ocp.args.Composite(
            **{k: ocp.args.StandardRestore(v) for k, v in abstract_state.items()}
        ),
    )
    return dict(composite)


def get_async_worker(max_workers=1):
    # async_worker.submit(Callable); async_worker.shutdown();
    async_worker = ThreadPoolExecutor(max_workers=max_workers)
    return async_worker


class ScalarLogger:
    """Buffered async scalar logger.

    Args:
        filepath: path to the output text file (created / appended).
        async_worker: a ``ThreadPoolExecutor`` used for the actual I/O.
    """

    def __init__(self, filepath, async_worker: ThreadPoolExecutor):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._worker = async_worker
        self._buf: list[str] = []
        self._buf_lock = threading.Lock()
        self._file_lock = threading.Lock()

    def log(self, step: int, value: float):
        """Appends ``step\\tvalue\\n`` lines to a file"""
        with self._buf_lock:
            self._buf.append(f"{step}\t{value:.6e}\n")

    def flush(self):
        """Swap buffer and write asynchronously, always call in the end"""
        with self._buf_lock:
            lines, self._buf = self._buf, []
        self._worker.submit(self._write, lines)

    def _write(self, lines: list[str]):
        with self._file_lock:
            with open(self.filepath, "a") as f:
                f.writelines(lines)
                f.flush()

    def plot(self):
        """Submit a plot of the log file to the async worker."""
        self._worker.submit(self._plot)

    def _plot(self):
        with self._file_lock:
            if not self.filepath.exists() or self.filepath.stat().st_size == 0:
                return
            data = np.loadtxt(self.filepath, delimiter="\t")
        steps, values = data[:, 0], data[:, 1]

        fig, ax = plt.subplots(layout="constrained")
        ax.plot(steps, values)
        ax.set_xlabel("step")
        ax.set_ylabel(self.filepath.stem)
        ax.set_yscale("log")
        fig.savefig(self.filepath.parent / f"{self.filepath.stem}.pdf")
        plt.close(fig)
