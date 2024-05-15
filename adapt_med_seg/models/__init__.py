from adapt_med_seg.models.segvol_base import  SegVolBase, SegVolLoraViT, SegVolLoraAll


MODELS = {
    "segvol_baseline": SegVolBase,
    "segvol_lora_vit": SegVolLoraViT,
    "segvol_lora_all": SegVolLoraAll,
}