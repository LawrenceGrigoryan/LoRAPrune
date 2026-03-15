from peft.peft_model import PeftModelForCausalLM
from .lora import LoraModel


class LoraPeftModelForCausalLM(PeftModelForCausalLM):
    def __init__(self, model, peft_config, adapter_name: str = "default"):
        super().__init__(model, peft_config, adapter_name)
        self.base_model = LoraModel(peft_config, model)
        self.active_adapter = adapter_name
        self.peft_config.__setattr__(adapter_name, peft_config)

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]


def get_peft_model(model, peft_config):
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    return LoraPeftModelForCausalLM(model, peft_config)
