import json
import netrc
import os
import re
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import PretrainedConfig

try:
    import wandb
except ImportError:
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
try:
    import swanlab
except ImportError:
    swanlab = None

def validate_report_to_args(parser, args):
    """
    only judge for wandb now
    """
    if args.report_to != "wandb":
        return

    if args.wandb_key is not None:
        return

    if "WANDB_API_KEY" in os.environ:
        args.wandb_key = os.environ["WANDB_API_KEY"]
        return

    try:
        netrc_path = os.path.expanduser("~/.netrc")
        if os.path.exists(netrc_path):
            netrc_file = netrc.netrc(netrc_path)
            if "api.wandb.ai" in netrc_file.hosts:
                login, account, password = netrc_file.authenticators("api.wandb.ai")
                if password:
                    args.wandb_key = password
                    return True
    except (FileNotFoundError, netrc.NetrcParseError):
        pass

    if args.wandb_key is None:
        parser.error(
            "When --wandb is enabled, you must provide a wandb API key via one of:\n"
            "  1. --wandb-key argument\n"
            "  2. WANDB_API_KEY environment variable\n"
            "  3. wandb login api-key"
        )

    elif args.report_to == "swanlab":
        if args.swanlab_key is not None:
            return
        if "SWANLAB_API_KEY" in os.environ:
            args.swanlab_key = os.environ["SWANLAB_API_KEY"]
            return
        if args.swanlab_key is None:
            parser.error(
                "In a distributed environment, when --report-to is 'swanlab', you must provide a swanlab API key via:\n"
                "  1. --swanlab-key argument\n"
                "  2. SWANLAB_API_KEY environment variable"
            )

class ExperimentTracker:
    """
    ExperimentTracker，support 'wandb', 'tensorboard', 'swanlab', or 'none'。
    """

    def __init__(self, args, output_dir: str):
        self.report_to = args.report_to
        self.rank = dist.get_rank()
        self.is_initialized = False

        if self.rank != 0:
            return

        if self.report_to == "wandb":
            if wandb is None:
                raise ImportError("wandb is not installed. Please install it with 'pip install wandb'")
            wandb.login(key=args.wandb_key)
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name
            )
            self.is_initialized = True
        elif self.report_to == "tensorboard":
            log_dir = os.path.join(output_dir, "runs")
            self.writer = SummaryWriter(log_dir=log_dir)
            self.is_initialized = True
        elif self.report_to == "swanlab":
            if swanlab is None:
                raise ImportError("swanlab is not installed. Please install it with 'pip install swanlab'")

            swanlab.login(api_key=args.swanlab_key)

            swanlog_dir = os.path.join(output_dir, "swanlog")
            os.makedirs(swanlog_dir, exist_ok=True)
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_name,
                config=vars(args),
                logdir=swanlog_dir
            )
            self.is_initialized = True
        elif self.report_to == "none":
            pass
        else:
            raise ValueError(f"Unsupported report_to type: {self.report_to}")

    def log(self, log_dict: dict, step: int = None):
        if self.rank != 0 or not self.is_initialized:
            return

        if self.report_to == "wandb":
            wandb.log(log_dict, step=step)
        elif self.report_to == "tensorboard":
            for key, value in log_dict.items():
                self.writer.add_scalar(key, value, global_step=step)
        elif self.report_to == "swanlab":
            swanlab.log(log_dict, step=step)
        elif self.report_to == "none":
            pass

    def close(self):
        if self.rank != 0 or not self.is_initialized:
            return

        if self.report_to == "wandb" and wandb.run:
            wandb.finish()
        elif self.report_to == "tensorboard":
            self.writer.close()
        elif self.report_to == "swanlab" and swanlab.is_running():
            swanlab.finish()

        self.is_initialized = False

@contextmanager
def rank_0_priority():
    rank = dist.get_rank()

    if rank == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield


@contextmanager
def default_torch_dtype(dtype: torch.dtype):
    current_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(current_dtype)


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def load_config_from_file(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    return PretrainedConfig.from_dict(config)


def print_with_rank(message):
    print(f"rank {dist.get_rank()}: {message}")


PREFIX_CHECKPOINT_DIR = "epoch"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"_(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder,
        max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])),
    )
