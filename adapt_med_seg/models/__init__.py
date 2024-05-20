from adapt_med_seg.models.segvol_base import  SegVolBase
from adapt_med_seg.models.segvol_lora import SegVolLORA
from adapt_med_seg.models.segvol_moe import SegVolMoE


MODELS = {
    "segvol_baseline": SegVolBase,
    "segvol_lora": SegVolLORA,
    "segvol_moe": SegVolMoE
}
