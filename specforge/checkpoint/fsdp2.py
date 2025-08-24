import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful


class SpecForgeStates(Stateful):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        scheduler_state_dict = self.scheduler.state_dict()
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        self.scheduler.load_state_dict(state_dict["scheduler"])


def save_checkpoint(model, optimizer, scheduler, path):
    state_dict = {"ckpt": SpecForgeStates(model, optimizer, scheduler)}
    dcp.save(state_dict, checkpoint_id=path)


def load_checkpoint(model, optimizer, scheduler, path):
    state_dict = {"ckpt": SpecForgeStates(model, optimizer, scheduler)}
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=path,
    )
