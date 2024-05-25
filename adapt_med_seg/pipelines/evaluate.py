from dataclasses import dataclass, field
import logging
from typing import Any

from tqdm import tqdm
import wandb

from adapt_med_seg.data.dataset import MedSegDataset, data_item_to_device
from SegVol.model_segvol_single import SegVolConfig
from adapt_med_seg.metrics import dice_score
from adapt_med_seg.utils.initializers import get_model
from adapt_med_seg.utils.average_meter import AverageMeter


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@dataclass
class EvaluateArgs:

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


class EvaluatePipeline:
    def __init__(self, **kwargs) -> None:
        self.model_name = kwargs["model_name"]
        self._use_wandb = kwargs["use_wandb"]
        self._wandb_project = kwargs["wandb_project"]
        self._dataset_path = kwargs["dataset_path"]
        self._modalities = kwargs["modalities"]
        self._device = kwargs["device"]
        self._batch_size = kwargs["batch_size"]
        self._max_len_samples = kwargs.get("max_len_test_samples", None)

        self._model = get_model(
            model_name=self.model_name,
            config=SegVolConfig(test_mode=True),
            kwargs=kwargs,
        )

        self._dataset = MedSegDataset(
            dataset_path=self._dataset_path,
            processor=self._model.processor,
            modalities=self._modalities,
            train=False,
        )
        self.dataset_id = self._dataset.dataset_number

    def run(self) -> dict[str, dict[str, Any]]:
        test_loader = self._dataset.get_test_dataloader(
            batch_size=self._batch_size, max_len_samples=self._max_len_samples
        )

        preds, labels = [], []

        results = {}

        avg_dice_score = AverageMeter()
        per_modality_scores = {
            modality_name: AverageMeter()
            for modality_name in self._dataset.modality_name2id.keys()
        }
        per_task_scores = {
            task: AverageMeter() for task in self._dataset.labels.values()
        }
        logger.info("Evaluating %s on dataset %s", self.model_name, self.dataset_id)

        if self._use_wandb:

            wandb.init(
                project=self._wandb_project,
                name=f"Evaluation_{self.model_name}_on_{self.dataset_id}",
            )
            wandb.config.update(
                {
                    "dataset": self.dataset_id,
                    "model": self.model_name,
                    "batch_size": self._batch_size,
                }
            )

        for batch in tqdm(
            test_loader,
            desc=f"Evaluating {self._dataset.name}",
            unit="batch",
        ):
            data_item, gt_npy, modality, task = batch
            data_item = data_item_to_device(data_item, self._model.device)

            # text prompt
            text_prompt = task[0]

            # point prompt
            point_prompt, point_prompt_map = self._model.processor.point_prompt_b(
                data_item["zoom_out_label"][0][0]
            )

            # bbox prompt
            bbox_prompt, bbox_prompt_map = self._model.processor.bbox_prompt_b(
                data_item["zoom_out_label"][0][0]
            )

            point_prompt = (
                point_prompt[0].to(self._model.device),
                point_prompt[1].to(self._model.device),
            )
            point_prompt_map = point_prompt_map.to(self._model.device)
            bbox_prompt = bbox_prompt.to(self._model.device)
            bbox_prompt_map = bbox_prompt_map.to(self._model.device)

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

            preds.append(pred[0][0])
            # labels.append(gt_npy)
            labels.append(data_item["label"][0][0])
            score = dice_score(
                preds[-1].to(self._model.device), labels[-1].to(self._model.device)
            )
            avg_dice_score.update(score)
            per_modality_scores[self._dataset.modality_id2name[modality[0]]].update(
                score
            )
            per_task_scores[task[0]].update(score)

        results = {
            "dice": float(avg_dice_score.avg),
            "per_modality_dice": {
                modality_name: float(score.avg)
                for modality_name, score in per_modality_scores.items()
            },
            "per_task_dice": {
                task: float(score.avg) for task, score in per_task_scores.items()
            },
        }

        if self._use_wandb:
            wandb.log(
                {
                    "dice_score": results["dice"],
                    "per_modality_dice": results["per_modality_dice"],
                }
            )
            wandb.finish()

        return results
