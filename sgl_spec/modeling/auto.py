import json

from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import LlamaConfig, PretrainedConfig

from .draft.llama3_eagle import LlamaForCausalLMEagle3


class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        LlamaConfig: LlamaForCausalLMEagle3,
    }

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        """
        This class method takes a configuration object and create its model based on the
        _model_mapping class variable.

        Args:
            config (PretrainedConfig): A configuration object.

        Returns:
            A model instance.
        """
        # get the model class from the
        _model_cls = cls._model_mapping[type(config)]
        return _model_cls(config)


class AutoEagle3TargetModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        LlamaConfig: [LlamaForCausalLMEagle3],
    }


class AutoDraftModelConfig:

    _config_mapping = {
        "LlamaForCausalLMEagle3": LlamaConfig,
    }

    @classmethod
    def from_file(cls, config_path: str):
        """
        This class method takes a configuration file path and create its configuration object based on the
        _config_mapping class variable.

        Args:
            config_path (str): A path to a configuration file.

        Returns:
            A configuration object.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        # check for architectures
        architectures = config.get("architectures", None)

        if architectures is None:
            raise ValueError("No architectures found in the config file")

        if len(architectures) != 1:
            raise ValueError("Only one architecture is supported")

        architecture = architectures[0]

        if architecture not in cls._config_mapping:
            raise ValueError(f"Architecture {architecture} not supported")

        return cls._config_mapping[architecture].from_dict(config)
