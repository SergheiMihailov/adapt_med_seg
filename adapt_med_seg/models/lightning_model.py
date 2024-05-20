from dataclasses import dataclass, field
from typing import Any
from SegVol.model_segvol_single import SegVolConfig
from adapt_med_seg.models import MODELS
from pytorch_lightning import LightningModule
import torch

from adapt_med_seg.data.dataset import MedSegDataset, data_item_to_device
from adapt_med_seg.metrics import dice_score

@dataclass
class TrainingArgs:
    use_wandb: bool = False
    model_name: str = "segvol_baseline"
    dataset_path: str = None,
    modalities: str = "CT",
    device: str = "cuda"
    batch_size: int = 1
    cls_idx: int = 0
    # text_prompt_template: str = "a photo of {}."
    seed: int = 42
    training_epochs: int = 10
    optimizer: Any = None
    test_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


class SegVolLightning(LightningModule):
    def __init__(self, train_args: TrainingArgs):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_args = train_args

        self.model_name = train_args.model_name
        self.dataset_path = train_args.dataset_path
        self.modalities = train_args.modalities

        self._cls_idx = train_args.cls_idx
        self._batch_size = train_args.batch_size
        self._use_wandb = train_args.use_wandb

        config = SegVolConfig(test_mode=train_args.test_mode)
        self._model = MODELS[train_args.model_name](config)

        self.processor = self._model.processor
        self.validation_step_outputs = []
        
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

        self.log("train_loss", loss.item())
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

        preds = pred[0][0]
        labels = data_item["label"][0][self._cls_idx]

        score = dice_score(preds, labels)
        self.validation_step_outputs.append(score)
        return score

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train()

    def on_validation_epoch_end(self):
        self.log(
            "val_dice_score",
            sum(self.validation_step_outputs) / len(self.validation_step_outputs),
        )
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda param: param.requires_grad, self.parameters()), lr=1e-4
        )