from adapt_med_seg.models.segvol_base import  SegVolBase
from adapt_med_seg.models.segvol_lora import SegVolLoRA
from adapt_med_seg.models.segvol_moe import SegVolMoE


MODELS = {
    "segvol_baseline": SegVolBase,
    "segvol_lora": SegVolLoRA,
    "segvol_moe": SegVolMoE
}
