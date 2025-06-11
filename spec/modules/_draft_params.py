import torch.nn as nn


def get_draft_params(model: nn.Module) -> dict[str, nn.Parameter]:
    """
    Return the subset of parameters from a model that correspond to draft layers.
    Returns all parameters whose names start with "draft.".

    Args:
        model (nn.Module): Instance of model class containing some draft params.

    Returns:
        dict[str, nn.Parameter]: the subset of model's state dict containing
        only draft parameters.
    """
    draft_params = {}
    for name, param in model.named_parameters():
        if name.startswith("draft."):
            draft_params[name] = param
    return draft_params
