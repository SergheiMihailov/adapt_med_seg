import torch


def dice_score(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Copied with minimal modifications from SegVol/model_segvol_single.py
    """
    assert preds.shape[0] == labels.shape[0], (
        "predict & target batch size don't match\n"
        + str(preds.shape)
        + str(labels.shape)
    )
    predict = preds.view(1, -1)
    target = labels.view(1, -1)

    predict = torch.sigmoid(predict)
    predict = torch.where(predict > 0.5, 1.0, 0.0)

    tp = torch.sum(torch.mul(predict, target))
    den = torch.sum(predict) + torch.sum(target) + 1
    dice = 2 * tp / den
    return dice
