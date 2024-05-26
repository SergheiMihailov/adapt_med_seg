from SegVol.model_segvol_single import SegVolConfig, SegVolModel

from adapt_med_seg.models.segvol_base import SegVolBase
from adapt_med_seg.models.segvol_lora import SegVolLoRA
from adapt_med_seg.models.segvol_moe import SegVolMoE
from adapt_med_seg.models.segvol_context_prior import SegVolContextPrior


def get_model(model_name: str, config, **kwargs):
    match model_name:
        case "segvol_baseline":
            model = SegVolBase(config)
        case "segvol_lora":
            model = SegVolLoRA(
                config,
                kwargs["target_modules"],
                kwargs["lora_r"],
                kwargs["lora_alpha"],
                kwargs["lora_dropout"],
                kwargs["train_only_vit"],
            )
        case "segvol_moe":
            model = SegVolMoE(
                config,
                kwargs["target_modules"],
                kwargs["lora_r"],
                kwargs["lora_alpha"],
                kwargs["lora_dropout"],
                kwargs["train_only_vit"],
            )
        case "segvol_context_prior":
            model = SegVolContextPrior(config)
        case _:
            raise ValueError(f"Model {model_name} not found.")
    return model
