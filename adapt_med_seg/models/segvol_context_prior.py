from typing import Mapping, Tuple, Type
import torch
from peft import PeftConfig, get_peft_model, PeftModel
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

@dataclass
class ContextPriorPoolConfig:
    tasks: list[str]
    modalities: list[str]
    embed_dim: int = 64

class ContextPriorPool:
    def __init__(self, tasks: list[str], modalities: list[str], embed_dim: int = 64):
        self.tasks = tasks
        self.modalities = modalities
        self.embed_dim = embed_dim

        self.task_prior_embeddings = nn.ModuleDict(
            {
                task: nn.Embedding(2, embed_dim)
                for task in self.tasks
            }
        )

        self.modality_prior_embeddings = nn.ModuleDict(
            {
                modality: nn.Embedding(2, embed_dim)
                for modality in self.modalities
            }
        )

    def get_task_prior(self, task: str):
        return self.task_prior_embeddings[task]

    def get_modality_prior(self, modality: str):
        return self.modality_prior_embeddings[modality]


class ContextPooledPromptEncoder(SamPromptEncoder):
    """
        Reimplement the PromptEncoder used in both SegVol and SAM
        to incorporate context priors.
    """

    def __init__(self,
                 context_prior_pool: ContextPriorPool,
                 embed_dim: int,
                 image_embedding_size: Tuple[int, int, int],
                 input_image_size: Tuple[int, int, int],
                 mask_in_chans: int,
                 activation: nn.Module = nn.GELU,
                 use_group_embeddings: bool = False) -> None:
        super().__init__(embed_dim, image_embedding_size,
                         input_image_size, mask_in_chans, activation)
        self.context_prior_pool = context_prior_pool
        # addition: group embeddings to better distinguish between different types of prompts
        self.use_group_embeddings = use_group_embeddings
        if self.use_group_embeddings:
            # there are 6 groups: point, box, mask, text, task, modality
            self.group_embeddings = nn.Embedding(6, embed_dim)
        else:
            # zeros
            self.group_embeddings = torch.zeros(1, embed_dim, requires_grad=False)

    def forward(self,
                points: Tuple[Tensor] | None = None,
                boxes: Tensor | None = None,
                masks: Tensor | None = None,
                text_embedding: Tensor | None = None,
                modality: str | None = None,
                task: str | None = None) -> Tuple[Tensor]:

        bs = self._get_batch_size(points, boxes, masks, text_embedding)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(
                coords, labels, pad=(boxes is None))
            point_embeddings = point_embeddings + self.group_embeddings[0]
            sparse_embeddings = torch.cat(
                [sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            box_embeddings = box_embeddings + self.group_embeddings[1]
            sparse_embeddings = torch.cat(
                [sparse_embeddings, box_embeddings], dim=1)

        if text_embedding is not None:
            text_embedding = text_embedding + self.group_embeddings[3]
            sparse_embeddings = torch.cat(
                [sparse_embeddings, text_embedding.unsqueeze(dim=1)], dim=1
            )

        if task is not None:
            task_prior = self.context_prior_pool.get_task_prior(task)
            task_prior = task_prior + self.group_embeddings[4]
            task_prior = task_prior.unsqueeze(dim=0).expand(bs, -1, -1)
            sparse_embeddings = torch.cat(
                [sparse_embeddings, task_prior], dim=1
            )

        if modality is not None:
            modality_prior = self.context_prior_pool.get_modality_prior(modality)
            modality_prior = modality_prior + self.group_embeddings[5]
            modality_prior = modality_prior.unsqueeze(dim=0).expand(bs, -1, -1)
            sparse_embeddings = torch.cat(
                [sparse_embeddings, modality_prior], dim=1
            )

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

    def load_from_pretrained(self, state_dict: Mapping[str, Tensor]) -> None:
        return super().load_state_dict(state_dict)

class SegVolContextPrior(SegVolModel):
    """
    SegVol model + using context priors as suggested in http://arxiv.org/abs/2103.00020
    """

    def __init__(self, config: SegVolConfig,
                 context_prior_pool_config: ContextPriorPoolConfig,
                 peft_config: PeftConfig):
        super().__init__(config)

        self.model: SegVol = AutoModel.from_pretrained(
            "BAAI/SegVol", trust_remote_code=True, test_mode=config.test_mode
        ).model

        print(f"keys: {self.model.keys()}")

        clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        self.model.text_encoder.tokenizer = clip_tokenizer

        self.processor = SegVolProcessor(spatial_size=self.config.spatial_size)

        # Context prior:
        self.context_prior_pool = ContextPriorPool(
            tasks=context_prior_pool_config.tasks,
            modalities=context_prior_pool_config.modalities,
            embed_dim=context_prior_pool_config.embed_dim,
        )
        # initialize a pooled prompt encoder and replace the original one
        # while keeping the pre-trained weights
        pooled_prompt_encoder = ContextPooledPromptEncoder(
            # addition: pass the context prior pool
            context_prior_pool=self.context_prior_pool,
            embed_dim=self.model.text_encoder.embed_dim,
            image_embedding_size=self.model.text_encoder.image_embedding_size,
            input_image_size=self.model.text_encoder.input_image_size,
            mask_in_chans=self.model.text_encoder.mask_in_chans,
            activation=self.model.text_encoder.activation,
            # addition: use group embeddings
            use_group_embeddings=self.config.use_group_embeddings
        )
        # load the pre-trained weights
        pooled_prompt_encoder.load_from_pretrained(self.model.text_encoder.state_dict())
        # replace the original prompt encoder
        self.model.prompt_encoder = pooled_prompt_encoder

        # PEFT
        self.model: PeftModel = get_peft_model(
            self.model, peft_config
        )
        logger.debug("SegVolContextPrior initialized.\n%s", self.model)

    def eval(self) -> Self:
        self.model.train(False)

    def train(self, mode=True) -> Self:
        self.model.train(mode)
        self.model.test_mode = not mode
        self.config.test_mode = not mode

    def to_cpu(self) -> Self:
        self.model.to("cpu")

    def to_cuda(self) -> Self:
        self.model.to("cuda")

    def to_mps(self) -> Self:
        self.model.to("mps")
