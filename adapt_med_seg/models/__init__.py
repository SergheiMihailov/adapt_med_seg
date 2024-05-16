from adapt_med_seg.models.segvol_base import  SegVolBase, SegVolLoraViT, SegVolLoraAll
from adapt_med_seg.models.segvol_lora import SegVolLORA


MODELS = {
    "segvol_baseline": SegVolBase,
    "segvol_lora": SegVolLORA,
}
