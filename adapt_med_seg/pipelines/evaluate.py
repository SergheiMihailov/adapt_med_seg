from dataclasses import dataclass, field
import logging
import torch
from typing import Any

from tqdm import tqdm
import wandb

from adapt_med_seg.data.dataset import MedSegDataset, data_item_to_device
from SegVol.model_segvol_single import SegVolConfig, generate_box, build_binary_cube
from adapt_med_seg.metrics import dice_score
from adapt_med_seg.utils.initializers import get_model
from adapt_med_seg.utils.average_meter import AverageMeter
import pytorch_lightning as pl
from adapt_med_seg.models.lightning_model import SegVolLightning


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
        self._prompt_types = kwargs["prompt_types"]
        self._perturb_bbox = kwargs["perturb_bbox"] # default=None
        self._checkpoint_path = kwargs.get("ckpt_path", None)

        # self._max_train_samples = kwargs.get("max_train_samples", None)
        # self._max_val_samples = kwargs.get("max_val_samples", None)
        self._max_test_samples = kwargs.get("max_test_samples", None)

        if self._checkpoint_path:
            _ = kwargs.pop("model_name")
            _ = kwargs.pop("modalities")
            self._model = SegVolLightning.load_from_checkpoint(self._checkpoint_path, self.model_name, self._modalities, True, **kwargs)
            self._model.eval()  
            logger.info(f"Loaded model checkpoint from {self._checkpoint_path}")

        self._dataset = MedSegDataset(
            dataset_path=self._dataset_path,
            processor=self._model.processor,
            modalities=self._modalities,
            train=False,
            max_train_samples=None, # never train in eval loop
            max_val_samples=None, # never validate in eval loop
            max_test_samples=self._max_test_samples,
        )
        self.dataset_id = self._dataset.dataset_number

    def run(self) -> dict[str, dict[str, Any]]:
        test_loader = self._dataset.get_test_dataloader(batch_size=self._batch_size)

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
        modality_counts = {modality_name: 0 for modality_name in per_modality_scores.keys()}

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
                    "prompt_types": self._prompt_types,
                    "perturb_bbox": self._perturb_bbox,
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
            text_prompt = None
            if "text" in self._prompt_types:
                text_prompt = task[0]
                logger.debug("Using text prompt %s" % text_prompt)

            # point prompt
            point_prompt, point_prompt_map = None, None
            if "point" in self._prompt_types:
                point_prompt, point_prompt_map = self._model.processor.point_prompt_b(
                    data_item["zoom_out_label"][0][0]
                )
                logger.debug("Using point prompt with shapes %s, %s (prompt) and %s (map)" %
                             (str(point_prompt[0].shape),
                              str(point_prompt[1].shape),
                              str(point_prompt_map.shape)))

            # bbox prompt
            bbox_prompt, bbox_prompt_map = None, None
            if "bbox" in self._prompt_types:
                bbox_prompt, bbox_prompt_map = self.perturbed_bbox_prompt_b(
                    data_item["zoom_out_label"][0][0]
                )
                logger.debug("Using bbox prompt with shapes %s (prompt) and %s (map)" %
                             (str(bbox_prompt.shape), str(bbox_prompt_map.shape)))

            point_prompt_group = None
            if point_prompt is not None and point_prompt_map is not None:
                point_prompt = (
                    point_prompt[0].to(self._model.device),
                    point_prompt[1].to(self._model.device),
                )
                point_prompt_map = point_prompt_map.to(self._model.device)
                point_prompt_group = (point_prompt, point_prompt_map)

            bbox_prompt_group = None
            if bbox_prompt is not None and bbox_prompt_map is not None:
                bbox_prompt = bbox_prompt.to(self._model.device)
                bbox_prompt_map = bbox_prompt_map.to(self._model.device)
                bbox_prompt_group = (bbox_prompt, bbox_prompt_map)

            self._model.model.test_mode = True
            pred = self._model.forward_test(
                image=data_item["image"],
                zoomed_image=data_item["zoom_out_image"],
                point_prompt_group=point_prompt_group,
                bbox_prompt_group=bbox_prompt_group,
                text_prompt=text_prompt,
                modality=self._dataset.modality_id2name[modality[0]],
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
            modality_counts[self._dataset.modality_id2name[modality[0]]] += 1


        results = {
            "dice": float(avg_dice_score.avg),
            "per_modality_dice": {
                modality_name: float(score.avg)
                for modality_name, score in per_modality_scores.items()
            },
            "per_task_dice": {
                task: float(score.avg) for task, score in per_task_scores.items()
            },
            "modality_counts": modality_counts,
        }

        if self._use_wandb:
            wandb.log(
                {
                    "dice_score": results["dice"],
                    "per_modality_dice": results["per_modality_dice"],
                    "per_task_dice": results["per_task_dice"],
                    "modality_counts": results["modality_counts"],
                }
            )
            wandb.finish()

        return results

    def perturbed_bbox_prompt_b(self, label_single_resize, device='cpu') -> torch.Tensor:
        """
        Reimplement `SegVol.processor.bbox_prompt_b` with random translation.
        Uses the `self._perturb_bbox` attribute to apply random translation.

        See `SegVol.model_segvol_single.SegVolProcessor.bbox_prompt_b` for more details.
        """
        box_single = generate_box(
            label_single_resize,
            bbox_shift=self._perturb_bbox # apply perturbation if requested
        ).unsqueeze(0).float().to(device)
        binary_cube_resize = (
            build_binary_cube(box_single, binary_cube_shape=label_single_resize.shape)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return box_single, binary_cube_resize
