import yaml
import argparse
from pathlib import Path
from datetime import datetime
from shutil import copyfile


class CMDArgsParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument(
            "--config", "--cf", required=True, type=str, help="Path to YAML config"
        )
        self.add_argument("--target", "-t", required=True, help="target in YAML config")
        self.add_argument(
            "--log",
            action=argparse.BooleanOptionalAction,
            default=True,
        )


def load_config():
    parser = CMDArgsParser()
    args = parser.parse_args()

    # Load YAML
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f).get(args.target)

    if config_dict is None:
        raise ValueError(f"Target '{args.target}' not found in config.")

    # Override YAML with non-None CLI args
    for key, value in vars(args).items():
        if value is not None:
            config_dict[key] = value

    return config_dict


# log_dir = Path(f"./").absolute() / config['base_dir'] / config['name'] \
#     / (datetime.now().strftime('%Y%m%d-%H%M%S') + "_" + config['version'])
# log_dir.mkdir(parents=True, exist_ok=True)