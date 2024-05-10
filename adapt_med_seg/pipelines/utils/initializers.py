from SegVol.model_segvol_single import SegVolConfig, SegVolModel
from adapt_med_seg.models import MODELS


def intialize_model(model_name: str, config: SegVolConfig, device: str) -> SegVolModel:
    model = MODELS[model_name](config)
    if device == "cuda":
        model.to_cuda()
    elif device == "mps":
        model.to_mps()
    else:
        model.to_cpu()

    model.eval()
    return model
