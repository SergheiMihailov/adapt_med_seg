from SegVol.model_segvol_single import SegVolConfig
from pytorch_lightning import LightningModule
import torch

from adapt_med_seg.data.dataset import MedSegDataset, data_item_to_device
from adapt_med_seg.metrics import dice_score
from adapt_med_seg.models.segvol_base import SegVolBase
from adapt_med_seg.models.segvol_lora import SegVolLoRA
from adapt_med_seg.models.segvol_moe import SegVolMoE
from adapt_med_seg.utils.initializers import get_model


class SegVolLightning(LightningModule):
    def __init__(
        self, model_name: str, modalities: list[str], test_mode: bool = False, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.modalities = modalities

        config = SegVolConfig(test_mode=test_mode)
        self._model = get_model(model_name, config, **kwargs)

        self.processor = self._model.processor

    def on_fit_start(self) -> None:
        if not hasattr(self, "_dataset"):
            raise ValueError("Dataset not set. Call set_dataset() before training.")
        return super().on_fit_start()

    def set_dataset(self, dataset: MedSegDataset):
        """Set the dataset for the training pipeline. This method should be called before training. If used in evaluation, you must supply the class index."""
        self._dataset = dataset

    def training_step(self, batch, batch_idx):
        return None
        data_item, gt_npy, modality, task = batch
        data_item["image"] = data_item["image"].to(self.device)

        modality = self._dataset.modality_id2name[modality[0]]
        # this is a mask ground truth
        gt_label = data_item["label"][0].to(self.device)

        tasks = task

        loss = self._model.forward_train(
            image=data_item["image"],
            tasks=tasks,
            train_labels=gt_label,
            modality=modality,
        )

        self.log("train_loss", loss.item(), prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data_item, gt_npy, modality, task = batch

        data_item = data_item_to_device(data_item, self.device)
        modality = self._dataset.modality_id2name[modality[0]]
        # text prompt
        text_prompt = task

        # point prompt
        point_prompt, point_prompt_map = self._model.processor.point_prompt_b(
            data_item["zoom_out_label"][0][0]
        )

        # bbox prompt
        bbox_prompt, bbox_prompt_map = self._model.processor.bbox_prompt_b(
            data_item["zoom_out_label"][0][0]
        )

        point_prompt = (
            point_prompt[0].to(self.device),  # point
            point_prompt[1].to(self.device),  # point label
        )
        point_prompt_map = point_prompt_map.to(self.device)
        bbox_prompt = bbox_prompt.to(self.device)
        bbox_prompt_map = bbox_prompt_map.to(self.device)

        print(f"[point_prompt, point_prompt_map]: {[point_prompt, point_prompt_map]}")

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
        labels = data_item["label"][0][0].to(self.device)

        preds = preds.to(self.device)
        labels = labels.to(self.device)

        score = dice_score(preds, labels)
        self.log("val_dice_score", score, prog_bar=True, on_epoch=True)

        return score

    def test_step(self, batch, batch_idx):
        # TODO: Implement test_step and inference_step
        return self.validation_step(batch, batch_idx)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train()
        self._dataset._train = True

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._dataset._train = False

    def configure_optimizers(self):
        lr = getattr(self.hparams, "lr", 5e-5)
        betas = getattr(self.hparams, "betas", (0.9, 0.999))
        eps = getattr(self.hparams, "eps", 1e-8)
        optimizer = torch.optim.AdamW(
            filter(lambda param: param.requires_grad, self.parameters()),
            lr=lr,
            betas=betas,
            eps=eps,
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return optimizer
