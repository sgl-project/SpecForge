import os
from omegaconf import OmegaConf


def load_config(config_path: str):
    """
    Load the config from the given path. 

    Args:
        config_path (str): The path to the config file.
    """
    config = OmegaConf.load(config_path)
    base_config_path = config.base

    # check if base_config_path is a relative path
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(os.path.dirname(config_path), base_config_path)
    base_config = OmegaConf.load(base_config_path)

    # merge config
    config = OmegaConf.merge(base_config, config)
    return config


def build_optimizer(model, config):
    """
    Build the optimizer from the config.
    """
    from torch import optim
    optim_cls = getattr(optim, config.type)
    cfg_copy = config.copy()
    cfg_copy.pop('type')
    return optim_cls(model.parameters(), **cfg_copy)

def build_criterion(config):
    """
    Build the criterion from the config.
    """
    from torch import nn
    criterion_cls = getattr(nn, config.type)
    cfg_copy = config.copy()
    cfg_copy.pop('type')
    return criterion_cls(**cfg_copy)

def build_model(config):
    """
    Build the model from the config.
    """
    # separate the model type and config kwargs
    import sgl_eagle.modeling as modeling
    cfg_copy = config.copy()
    model_type = cfg_copy.pop('type')
    model_cls = getattr(modeling, model_type)

    # get the config class
    cfg_cls = model_cls.config_class
    model_cfg = cfg_cls(**cfg_copy)
    return model_cls(model_cfg)