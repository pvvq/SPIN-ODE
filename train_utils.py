import yaml
import argparse


class DummyLogger:
    def add_scalar(self, *args, **kwargs): pass
    def add_figure(self, *args, **kwargs): pass
    def add_text(self, *args, **kwargs): pass

class DummyCheckpointer:
    def save(self, *args, **kwargs): pass


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