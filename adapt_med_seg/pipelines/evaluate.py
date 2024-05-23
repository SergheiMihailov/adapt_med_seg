from dataclasses import dataclass, field
import logging
from typing import Any

from tqdm import tqdm
import wandb

from adapt_med_seg.data.dataset import MedSegDataset, data_item_to_device
from SegVol.model_segvol_single import SegVolConfig
from adapt_med_seg.metrics import dice_score
from adapt_med_seg.pipelines.utils.initializers import intialize_model
from adapt_med_seg.utils.average_meter import AverageMeter


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@dataclass
class EvaluateArgs:
    use_wandb: bool = False
    model_name: str = "segvol_baseline"
    dataset_path: str = None
    modalities: str = field(default_factory=lambda: ["CT"])
    device: str = "cuda"
    batch_size: int = 1
    # text_prompt_template: str = "a photo of {}."
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


class EvaluatePipeline:
    def __init__(
        self,
        evaluate_args: EvaluateArgs,
    ) -> None:
        self._model = intialize_model(
            model_name=evaluate_args.model_name,
            config=SegVolConfig(test_mode=True),
            device=evaluate_args.device,
        )

        self._dataset = MedSegDataset(
            dataset_path=evaluate_args.dataset_path,
            processor=self._model.processor,
            modalities=evaluate_args.modalities,
            train=False,
        )

        self.model_name = evaluate_args.model_name
        self.modalities = evaluate_args.modalities
        self.dataset_id = self._dataset.dataset_number

        self._batch_size = evaluate_args.batch_size
        self._use_wandb = evaluate_args.use_wandb

    def run(self) -> dict[str, dict[str, Any]]:
        test_loader = self._dataset.get_test_dataloader(
            batch_size=self._batch_size)

        preds, labels = [], []

        results = {}

        avg_dice_score = AverageMeter()

        logger.info("Evaluating %s on dataset %s",
                    self.model_name, self.dataset_id)

        if self._use_wandb:

            wandb.init(
                project="dl2_g33",
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
            text_prompt = task

            # point prompt
            point_prompt, point_prompt_map = self._model.processor.point_prompt_b(
                data_item["zoom_out_label"][0]
            )

            # bbox prompt
            bbox_prompt, bbox_prompt_map = self._model.processor.bbox_prompt_b(
                data_item["zoom_out_label"][0]
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
            labels.append(data_item["label"][0])

            avg_dice_score.update(dice_score(
                preds[-1].to(self._model.device), labels[-1].to(self._model.device)))

        results = {"dice": avg_dice_score.avg}

        if self._use_wandb:
            wandb.log({"dice_score": results["dice"]})
            wandb.finish()

        return results
