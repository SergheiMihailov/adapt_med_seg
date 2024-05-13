from dataclasses import dataclass
import logging
from typing import Any

import torch
from tqdm import tqdm
import wandb

from adapt_med_seg.data.dataset import MedSegDataset, data_item_to_device
from SegVol.model_segvol_single import SegVolConfig
from adapt_med_seg.metrics import dice_score
from adapt_med_seg.models import MODELS
from adapt_med_seg.pipelines.utils.initializers import intialize_model
from adapt_med_seg.utils.average_meter import AverageMeter
import pytorch_lightning as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingArgs:
    use_wandb: bool = False
    model_name: str = "segvol_baseline"
    dataset_number: int = 0
    device: str = "cuda"
    batch_size: int = 1
    cls_idx: int = 0
    # text_prompt_template: str = "a photo of {}."
    seed: int = 42
    training_epochs: int = 10
    optimizer: Any = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


# class TrainingPipeline:
#     def __init__(
#         self,
#         training_args: TrainingArgs,
#     ) -> None:
#         self._model = intialize_model(
#             model_name=training_args.model_name,
#             config=SegVolConfig(test_mode=False),
#             device=training_args.device,
#         )

#         self._dataset = MedSegDataset(
#             dataset_number=training_args.dataset_number,
#             processor=self._model.processor,
#             train=True,
#         )

#         self.model_name = training_args.model_name
#         self.dataset_number = training_args.dataset_number

#         self._cls_idx = training_args.cls_idx
#         self._batch_size = training_args.batch_size
#         self._use_wandb = training_args.use_wandb
#         self.training_epochs = training_args.training_epochs

#         self.optimizer = training_args.optimizer or torch.optim.Adam(filter(lambda param: param.requires_grad, self._model.parameters()), lr=1e-4)

#     def run(self) -> dict[str, dict[str, Any]]:
#         device = self._model.device
#         train_loader, val_loader = self._dataset.get_train_val_dataloaders(1/9, 1, 42)

#         results = {}
#         train_loss = AverageMeter()

#         logger.info("Training %s on dataset %s", self.model_name, self.dataset_number)

#         if self._use_wandb:

#             wandb.init(
#                 project="dl2_g33",
#                 name=f"Training_{self.model_name}_on_{self.dataset_number}",
#             )
#             wandb.config.update(
#                 {
#                     "dataset": self.dataset_number,
#                     "model": self.model_name,
#                     "cls_idx": self._cls_idx,
#                     "batch_size": self._batch_size,
#                 }
#             )


#         avg_dice_score = AverageMeter()
#         for epoch in range(self.training_epochs):
#             logging.info(f"Epoch {epoch + 1}/{self.training_epochs}")

#             self._model.train()
#             self._model.model.test_mode = False
#             for batch in tqdm(
#                 train_loader,
#                 desc=f"Training on {self._dataset.name}",
#             ):
#                 data_item, gt_npy = batch
#                 data_item["image"] = data_item["image"].to(device)

#                 # this is a mask ground truth
#                 gt_label = data_item["label"].to(device)

#                 train_organs = self._dataset.labels[self._cls_idx]
#                 train_labels = gt_label[:, self._cls_idx].to(device)

#                 loss = self._model.forward_train(
#                     image=data_item["image"],
#                     train_organs=train_organs,
#                     train_labels=train_labels,
#                 )

#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#                 train_loss.update(loss.item())

#             ############################
#             # Evaluation
#             ############################
#             self._model.eval()
#             self._model.model.test_mode = True

#             _avg_dice_score = AverageMeter()
#             preds, labels = [], []
#             for batch in tqdm(
#                 val_loader,
#                 desc=f"Evaluating {self._dataset.name}",
#             ):
#                 data_item, gt_npy = batch
#                 data_item = data_item_to_device(data_item, device)

#                 # text prompt
#                 text_prompt = [self._dataset.labels[self._cls_idx]]

#                 # point prompt
#                 point_prompt, point_prompt_map = self._model.processor.point_prompt_b(
#                     data_item["zoom_out_label"][0][self._cls_idx]
#                 )

#                 # bbox prompt
#                 bbox_prompt, bbox_prompt_map = self._model.processor.bbox_prompt_b(
#                     data_item["zoom_out_label"][0][self._cls_idx]
#                 )

#                 point_prompt = (
#                     point_prompt[0].to(device), # point
#                     point_prompt[1].to(device), # point label
#                 )
#                 point_prompt_map = point_prompt_map.to(device)
#                 bbox_prompt = bbox_prompt.to(device)
#                 bbox_prompt_map = bbox_prompt_map.to(device)

#                 pred = self._model.forward_test(
#                     image=data_item["image"],
#                     zoomed_image=data_item["zoom_out_image"],
#                     point_prompt_group=[point_prompt, point_prompt_map],
#                     bbox_prompt_group=(
#                         None if point_prompt else [bbox_prompt, bbox_prompt_map]
#                     ),
#                     text_prompt=text_prompt,
#                     use_zoom=True,
#                 )

#                 preds.append(pred[0][0])
#                 # labels.append(gt_npy)
#                 labels.append(data_item["label"][0][self._cls_idx])

#                 _avg_dice_score.update(dice_score(preds[-1].to(device), labels[-1].to(device)))
#             ############################
#             avg_dice_score.update(_avg_dice_score.avg)
#             logging.info(f"Dice score: {avg_dice_score.avg}")

#             if self._use_wandb:
#                 wandb.log({"dice_score": avg_dice_score.avg})

#         if self._use_wandb:
#             wandb.finish()
#         wandb.finish()

#         return results


class SegVolLightning(pl.LightningModule):
    def __init__(
        self,
        model_name="segvol_baseline",
        dataset_number=0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._model = MODELS[model_name](SegVolConfig(test_mode=False))
        self.processor = self._model.processor

        self.dataset_number = dataset_number
        self.validation_step_outputs = []

    def set_dataset(self, dataset: MedSegDataset, cls_idx: int = 0):
        """Set the dataset for the training pipeline. This method should be called before training. If used in evaluation, you must supply the class index."""
        self._dataset = dataset
        self._cls_idx = cls_idx

    def training_step(self, batch, batch_idx):
        data_item, gt_npy, modality = batch
        data_item["image"] = data_item["image"].to(self.device)

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
        )

        preds = pred[0][0]
        labels = data_item["label"][0][self._cls_idx]

        score = dice_score(preds, labels)
        self.validation_step_outputs.append(score)
        return score

    def train(self, mode=True):
        super().train(mode)
        self._model.model.test_mode = False
        self._model.config.test_mode = False

    def eval(self):
        super().eval()
        self._model.model.test_mode = True
        self._model.config.test_mode = True

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

    def get_dataloaders(self):
        return self._dataset.get_train_val_dataloaders(1 / 9, 1, 42)

    def train_dataloader(self):
        return self._dataset.get_train_val_dataloaders(1 / 9, 1, 42)[0]

    def val_dataloader(self):
        return self._dataset.get_train_val_dataloaders(1 / 9, 1, 42)[1]
