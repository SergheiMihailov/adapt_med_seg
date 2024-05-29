from dataclasses import dataclass, field
import logging
import torch
from typing import Any
import os
import yaml

from tqdm import tqdm
from collections import defaultdict
import wandb
import json
import os

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
        self._log_every_n_steps = kwargs.get("log_every_n_steps", 10) # log to wandb every n steps
        self._log_dir = kwargs.get("log_dir", None)
        if self._log_dir is not None:
            try:
                os.makedirs(self._log_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create log dir: {e}")
                self._log_dir = None

        # self._max_train_samples = kwargs.get("max_train_samples", None)
        # self._max_val_samples = kwargs.get("max_val_samples", None)
        self._max_test_samples = kwargs.get("max_test_samples", None)

        _ = kwargs.pop("model_name")
        _ = kwargs.pop("modalities")
        # this could be even empty, doesn't matter
        tasks = ["unknown", "duodenum", "prostate", "colon cancer", "pancreas", "Edema", "Non-Contrast-Enhancing Tumor Core", "tumour", "arota",
                     "Enhancing Tumor", "bladder", "esophagus", "prostate/uterus", "right adrenal gland", "gall bladder", "left adrenal gland",
                     "postcava", "stomach",'liver', 'spleen', 'right kidney', 'left kidney']
        # break the cyclic dependency. this is not the real model but just to get the processor
        self._model = SegVolLightning(self.model_name, self._modalities, tasks, True, **kwargs)

        self._dataset = MedSegDataset(
            dataset_path=self._dataset_path,
            processor=self._model.processor,
            modalities=self._modalities,
            train=False,
            max_train_samples=None, # never train in eval loop
            max_val_samples=None, # never validate in eval loop
            max_test_samples=self._max_test_samples,
        )

        print("Modalities:", self._dataset.modalities)
        print("Tasks:", list(self._dataset.labels.values()))
        if self._checkpoint_path:
            # tasks = list(self._dataset.labels.values()) # this will bug out checkpoint loading if the eval dataset doesnt have a category that the training had
            # instead read from the hparams.yaml
            data = yaml.safe_load(os.path.join(self._checkpoint_path, "..", "..","hparams.yaml"))
            tasks = data['tasks']
            self._model = SegVolLightning.load_from_checkpoint(self._checkpoint_path, self.model_name, self._modalities, tasks, True, **kwargs)
            self._model.eval()  
            logger.info(f"Loaded model checkpoint from {self._checkpoint_path}")
            print("Loaded model checkpoint from", self._checkpoint_path)
        else:
            self._model = SegVolLightning(self.model_name, self._modalities, list(self._dataset.labels.values()), True, **kwargs)
            self._model.eval()

        self.dataset_id = self._dataset.dataset_number

    def run(self) -> dict[str, dict[str, Any]]:
        test_loader = self._dataset.get_test_dataloader(batch_size=self._batch_size)

        results = {}

        avg_dice_score = AverageMeter()
        per_dataset_modality_task_scores = {}
        per_dataset_modality_task_counts = {}
        for dataset_num in self._dataset.dataset_numbers:
            per_dataset_modality_task_scores[dataset_num] = {}
            per_dataset_modality_task_counts[dataset_num] = {}
            for modality in self._dataset._modalities:
                per_dataset_modality_task_scores[dataset_num][modality] = defaultdict(AverageMeter)
                per_dataset_modality_task_counts[dataset_num][modality] = defaultdict(int)

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

        for idx, batch in enumerate(tqdm(
            test_loader,
            desc=f"Evaluating {self._dataset.name}",
            unit="batch",
        )):
            data_item, gt_npy, modality, task = batch
            task = task[0]
            #print("task:", task)
            data_item = data_item_to_device(data_item, self._model.device)

            # hard-coding to bs=1 to avoid unnecessary bugs.
            # this is *not* what breaks batching in this poop storm of a codebase
            case_dataset_name = self._dataset.get_last_dataset_names(1)[0]
            case_modality = self._dataset.modality_id2name[modality[0]]

            # text prompt
            text_prompt = None
            if "text" in self._prompt_types:
                #if int(modality[0]) == 0:
                #    text_prompt = f"A computerized tomography of a {task}"
                #elif int(modality[0]) == 1:
                #    text_prompt = f"A magnetic resonance image (MRI) of a {task}"
                text_prompt = task # the SegVol model if the Promptencoder is not overriden, will attach to it "A computerized tomography of"
                # we want it to attach it to MRI too because it learnt on CT
                logger.debug("Using text prompt %s" % text_prompt)
                print("text_prompt:", text_prompt, "modality:", modality)

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

            self._model._model.test_mode = True
            pred = self._model._model.forward_test(
                image=data_item["image"],
                zoomed_image=data_item["zoom_out_image"],
                point_prompt_group=point_prompt_group,
                bbox_prompt_group=bbox_prompt_group,
                text_prompt=text_prompt,
                modality=case_modality,
                use_zoom=True,
            )

            score = dice_score(
                pred[0][0].to(self._model.device), data_item["label"][0][0].to(
                    self._model.device)
            )
            avg_dice_score.update(score)
            per_dataset_modality_task_scores[case_dataset_name][case_modality][task].update(score)
            per_dataset_modality_task_counts[case_dataset_name][case_modality][task] += 1

            # log to wandb every n steps
            if idx % self._log_every_n_steps == 0:
                results = {
                    "dice": float(avg_dice_score.avg),
                    "per_dataset_modality_task_dice": {
                        dataset: {
                            modality_name: {
                                task: float(score.avg)
                                for task, score in scores.items()
                            } for modality_name, scores in scores.items()
                        } for dataset, scores in per_dataset_modality_task_scores.items()
                    },
                    "per_dataset_modality_task_counts": per_dataset_modality_task_counts
                }
                # log to stdout and log dir if specified
                result_str = json.dumps({idx: results}, indent=4)
                if self._log_dir:
                    try:
                        with open(f"{self._log_dir}/results_{idx}.json", "w") as f:
                            f.write(result_str)
                    except Exception as e:
                        # avoid unnexessary bugs
                        logger.error(f"Failed to write results to log dir: {e}")
                print(result_str)
                # also to wandb if requested
                if self._use_wandb:
                    wandb.log(
                        {
                            "dice_score": results["dice"],
                            "per_dataset_modality_task_dice": \
                                results["per_dataset_modality_task_dice"],
                            "per_dataset_modality_task_counts": \
                                results["per_dataset_modality_task_counts"],
                        }
                    )

        results = {
            "dice": float(avg_dice_score.avg),
            "per_dataset_modality_task_dice": {
                dataset: {
                    modality_name: {
                        task: float(score.avg)
                        for task, score in scores.items()
                    } for modality_name, scores in scores.items()
                } for dataset, scores in per_dataset_modality_task_scores.items()
            },
            "per_dataset_modality_task_counts": per_dataset_modality_task_counts,
        }
        if self._use_wandb:
            wandb.log(
                {
                    "dice_score": results["dice"],
                    "per_dataset_modality_task_dice": results["per_dataset_modality_task_dice"],
                    "per_dataset_modality_task_counts": results["per_dataset_modality_task_counts"],
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
