from SegVol.model_segvol_single import SegVolConfig
from adapt_med_seg.models import MODELS
from pytorch_lightning import LightningModule
import torch

from adapt_med_seg.data.dataset import MedSegDataset, data_item_to_device
from adapt_med_seg.metrics import dice_score
from adapt_med_seg.models.segvol_base import SegVolBase
from adapt_med_seg.models.segvol_lora import SegVolLoRA

def get_model(model_name: str, config, **kwargs):
    
    match model_name:
        case "segvol_baseline":
            model = SegVolBase(config)
        case "segvol_lora":
            model = SegVolLoRA(config, **kwargs)
        case _:
            raise ValueError(f"Model {model_name} not found.")
    return model

class SegVolLightning(LightningModule):
    def __init__(self, model_name: str, modalities: list[str], use_wandb: bool, test_mode: bool = False, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.modalities = modalities
        self._use_wandb = use_wandb

        config = SegVolConfig(test_mode=test_mode)
        self._model = get_model(model_name, config, **kwargs)

        self.processor = self._model.processor
        self.validation_step_outputs = []
    
    def on_fit_start(self) -> None:
        if not hasattr(self, "_dataset") or not hasattr(self, "_cls_idx"):
            raise ValueError("Dataset not set. Call set_dataset() before training.")
        return super().on_fit_start()    
    
    def set_dataset(self, dataset: MedSegDataset, cls_idx: int = 0):
        """Set the dataset for the training pipeline. This method should be called before training. If used in evaluation, you must supply the class index."""
        self._dataset = dataset
        self._cls_idx = cls_idx

    def training_step(self, batch, batch_idx):
        data_item, gt_npy, modality = batch
        data_item["image"] = data_item["image"].to(self.device)

        modality = self._dataset.modality_id2name[modality[0]]
        # this is a mask ground truth
        gt_label = data_item["label"].to(self.device)
        # I think we need to handle multiple classes here?
        loss = None
        for cls_idx in range(len(self._dataset.labels)):
            train_organs = self._dataset.labels[cls_idx]
            train_labels = gt_label[:, cls_idx].to(self.device)

            _loss = self._model.forward_train(
                image=data_item["image"],
                train_organs=train_organs,
                train_labels=train_labels,
                modality=modality,
            )

            if loss is None:
                loss = _loss
            else:
                loss = loss + _loss

        self.log("train_loss", loss.item(), prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data_item, gt_npy, modality = batch
        data_item = data_item_to_device(data_item, self.device)
        modality = self._dataset.modality_id2name[modality[0]]
        # text prompt
        text_prompt = [self._dataset.labels[self._cls_idx]]

        # point prompt
        point_prompt, point_prompt_map = self._model.processor.point_prompt_b(
            data_item["zoom_out_label"][0][self._cls_idx]
        )

        # bbox prompt
        bbox_prompt, bbox_prompt_map = self._model.processor.bbox_prompt_b(
            data_item["zoom_out_label"][0][self._cls_idx]
        )

        point_prompt = (
            point_prompt[0].to(self.device),  # point
            point_prompt[1].to(self.device),  # point label
        )
        point_prompt_map = point_prompt_map.to(self.device)
        bbox_prompt = bbox_prompt.to(self.device)
        bbox_prompt_map = bbox_prompt_map.to(self.device)

        pred = self._model.forward_test(
            image=data_item["image"],
            zoomed_image=data_item["zoom_out_image"],
            point_prompt_group=[point_prompt, point_prompt_map],
            bbox_prompt_group=(
                None if point_prompt else [bbox_prompt, bbox_prompt_map]
            ),
            text_prompt=text_prompt,
            use_zoom=True,
            modality=modality,
        )

        preds = pred[0][0].to(self.device)
        labels = data_item["label"][0][self._cls_idx].to(self.device)

        score = dice_score(preds, labels)
        self.log("val_dice_score", score, prog_bar=True, on_epoch=True)
        return score

    def test_step(self, batch, batch_idx):
        # TODO: Implement test_step and inference_step, additionally, wandb logging
        return self.validation_step(batch, batch_idx)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train()

    def configure_optimizers(self):
        lr = getattr(self.hparams, "lr", 5e-5)
        betas = getattr(self.hparams, "betas", (0.9, 0.999))
        eps = getattr(self.hparams, "eps", 1e-8)
        optimizer = torch.optim.AdamW(
            filter(lambda param: param.requires_grad, self.parameters()), lr=lr, betas=betas, eps=eps
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return optimizer