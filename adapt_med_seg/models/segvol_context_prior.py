import numpy as np
from typing import Mapping, Tuple, Type
import torch
from peft import LoraConfig, get_peft_model, PeftModel
import re
import logging
from torch import Tensor, nn
from typing_extensions import Self
from transformers import AutoTokenizer, AutoModel
from SegVol.model_segvol_single import (
    SegVolConfig,
    SegVolModel,
    SegVolProcessor,
    SegVol,
    PromptEncoder as SamPromptEncoder,
)
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ContextPriorPool:
    def __init__(
        self,
        tasks: list[str],
        modalities: list[str],
        dtype: Type[torch.dtype],
        embed_dim: int = 64,
    ):
        self.tasks = tasks
        self.modalities = modalities
        self.embed_dim = embed_dim

        self.task_prior_embeddings = nn.ModuleDict(
            {task: torch.zeros(embed_dim, dtype=dtype) for task in self.tasks}
        )

        self.modality_prior_embeddings = nn.ModuleDict(
            {
                modality: torch.zeros(embed_dim, dtype=dtype)
                for modality in self.modalities
            }
        )

    def get_task_prior(self, task: str):
        return self.task_prior_embeddings[task]

    def get_modality_prior(self, modality: str):
        return self.modality_prior_embeddings[modality]

    def to(self, device):
        for prior in self.task_prior_embeddings.values():
            prior.to(device)
        for prior in self.modality_prior_embeddings.values():
            prior.to(device)


class ContextPooledPromptEncoder(SamPromptEncoder):
    """
    Reimplement the PromptEncoder used in both SegVol and SAM
    to incorporate context priors.
    """

    def __init__(
        self,
        context_prior_pool: ContextPriorPool,
        embed_dim: int,
        image_embedding_size: Tuple[int, int, int],
        input_image_size: Tuple[int, int, int],
        mask_in_chans: int,
        activation: nn.Module = nn.GELU,
        use_group_embeddings: bool = False,
    ) -> None:
        super().__init__(
            embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation
        )
        self.context_prior_pool = context_prior_pool
        # addition: group embeddings to better distinguish between different types of prompts
        self.use_group_embeddings = use_group_embeddings
        if self.use_group_embeddings:
            # there are 6 groups: point, box, mask, text, task, modality
            self.group_embeddings = nn.Parameter(torch.zeros(6, embed_dim))
        else:
            # zeros
            self.group_embeddings = torch.zeros(6, embed_dim, requires_grad=False)

    def forward(
        self,
        points: Tuple[Tensor] | None = None,
        boxes: Tensor | None = None,
        masks: Tensor | None = None,
        text_embedding: Tensor | None = None,
        modality: str | None = None,
        task: str | None = None,
    ) -> Tuple[Tensor]:

        bs = self._get_batch_size(points, boxes, masks, text_embedding)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            point_embeddings = point_embeddings + self.group_embeddings[0]
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            box_embeddings = box_embeddings + self.group_embeddings[1]
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if text_embedding is not None:
            text_embedding = text_embedding + self.group_embeddings[3]
            sparse_embeddings = torch.cat(
                [sparse_embeddings, text_embedding.unsqueeze(dim=1)], dim=1
            )

        if task is not None:
            task_prior = self.context_prior_pool.get_task_prior(task)
            task_prior = task_prior + self.group_embeddings[4]
            task_prior = task_prior.unsqueeze(dim=0).expand(bs, -1, -1)
            sparse_embeddings = torch.cat([sparse_embeddings, task_prior], dim=1)

        if modality is not None:
            modality_prior = self.context_prior_pool.get_modality_prior(modality)
            modality_prior = modality_prior + self.group_embeddings[5]
            modality_prior = modality_prior.unsqueeze(dim=0).expand(bs, -1, -1)
            sparse_embeddings = torch.cat([sparse_embeddings, modality_prior], dim=1)

        # keeping for compatibility, but SegVol actually does not use mask prompts
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(
                bs,
                -1,
                int(self.image_embedding_size[0]),
                int(self.image_embedding_size[1]),
                int(self.image_embedding_size[2]),
            )

        return sparse_embeddings, dense_embeddings

    # def load_from_pretrained(self, state_dict: Mapping[str, Tensor]) -> None:
    #     return super().load_state_dict(state_dict)
    def load_from_pretrained(self, state_dict: Mapping[str, Tensor]) -> None:
        # Get the current model's state dictionary
        current_state_dict = self.state_dict()

        # Filter the loaded state dictionary to only include keys that exist in the current model
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if k in current_state_dict
        }

        # Load the filtered state dictionary, setting strict=False to ignore any non-matching keys
        self.load_state_dict(filtered_state_dict, strict=False)


class SegVolContextPrior(SegVolModel):
    """
    SegVol model + using context priors as suggested in http://arxiv.org/abs/2103.00020
    """

    def __init__(self, config: SegVolConfig, **kwargs):
        super().__init__(config)

        self.model: SegVol = AutoModel.from_pretrained(
            "BAAI/SegVol", trust_remote_code=True, test_mode=config.test_mode
        ).model

        clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        self.model.text_encoder.tokenizer = clip_tokenizer

        self.processor = SegVolProcessor(spatial_size=self.config.spatial_size)

        embed_dim = 768

        # Context prior:
        self.context_prior_pool = ContextPriorPool(
            dtype=torch.float32,
            tasks=kwargs.get("tasks", []),
            modalities=kwargs.get("modalities", []),
            embed_dim=embed_dim,
        )

        patch_size = self.config.patch_size
        image_size = self.config.spatial_size

        image_embedding_size = [
            int(item) for item in (np.array(image_size) / np.array(patch_size))
        ]

        # initialize a pooled prompt encoder and replace the original one
        # while keeping the pre-trained weights
        pooled_prompt_encoder = ContextPooledPromptEncoder(
            # addition: pass the context prior pool
            context_prior_pool=self.context_prior_pool,
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=image_size,
            mask_in_chans=16,
            # activation=# self.model.text_encoder.activation,
            # addition: use group embeddings
            use_group_embeddings=True,
        )
        # load the pre-trained weights
        pooled_prompt_encoder.load_from_pretrained(
            self.model.prompt_encoder.state_dict()
        )
        # replace the original prompt encoder
        self.model.prompt_encoder = pooled_prompt_encoder

        self.model.prompt_encoder.train(True)

        # PEFT
        # peft_config = LoraConfig(
        #     target_modules=kwargs.get('target_modules', ["Conv2d", "LayerNorm2d"]),
        #     inference_mode=config.test_mode,
        #     r=kwargs.get('lora_r', 8),
        #     lora_alpha=kwargs.get('lora_alpha', 8),
        #     lora_dropout=kwargs.get('lora_dropout', 0.0)
        # )
        # self.model: PeftModel = get_peft_model(
        #     self.model, peft_config
        # )
        logger.debug("SegVolContextPrior initialized.\n%s", self.model)

    def eval(self) -> Self:
        self.model.train(False)

    def train(self, mode=True) -> Self:
        self.model.train(mode)
        self.model.test_mode = not mode
        self.config.test_mode = not mode

    def to_cpu(self) -> Self:
        self.model.to("cpu")
        self.context_prior_pool.to("cpu")

    def to_cuda(self) -> Self:
        self.model.to("cuda")
        self.context_prior_pool.to("cuda")

    def to_mps(self) -> Self:
        self.model.to("mps")
        self.context_prior_pool.to("mps")
