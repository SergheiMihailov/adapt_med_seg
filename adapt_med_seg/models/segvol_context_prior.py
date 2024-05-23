from typing import Mapping, Tuple, Type
import torch
from torch import Tensor, nn
from typing_extensions import Self

from SegVol.model_segvol_single import (
    SegVol,
    SegVolConfig,
    SegVolModel,
    SegVolProcessor,
    SegVol,
    PromptEncoder as SamPromptEncoder,
)
from dataclasses import dataclass


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
            {task: nn.Embedding(2, embed_dim) for task in self.tasks}
        )

        self.modality_prior_embeddings = nn.ModuleDict(
            {modality: nn.Embedding(2, embed_dim) for modality in self.modalities}
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
            self.group_embeddings = nn.Embedding(6, embed_dim)
        else:
            # zeros
            self.group_embeddings = torch.zeros(1, embed_dim, requires_grad=False)

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

    def load_from_pretrained(self, state_dict: Mapping[str, Tensor]) -> None:
        return super().load_state_dict(state_dict)

    def forward_decoder(
        self,
        image_embedding,
        img_shape,
        text=None,
        boxes=None,
        points=None,
        task_prior=None,
        modality_prior=None,
    ):
        device = image_embedding.device
        with torch.no_grad():
            if boxes is not None:
                if len(boxes.shape) == 2:
                    boxes = boxes[:, None, :]  # (B, 1, 6)
            if text is not None:
                text_embedding = self.pretrained_segvol.text_encoder(
                    text, device
                )  # (B, 768)
            else:
                text_embedding = None

        sparse_embeddings, dense_embeddings = self.pretrained_segvol.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
            text_embedding=text_embedding,
        )

        dense_pe = self.pretrained_segvol.prompt_encoder.get_dense_pe()

        # Low-res mask
        # mask_tokens.shape: torch.Size([1, 1, 32, 64, 64])
        # iou_tokens.shape: torch.Size([1, 1])

        # Mask tokens
        # mask_tokens.shape: torch.Size([1, 4, 768])
        # iou_tokens.shape: torch.Size([1, 768])

        # 2 possible approaches here:
        # 1) apply prototype to mask tokens
        # 2) apply prototype to low-res mask. But then, what is C' in this case?

        # Approach 1

        mask_tokens, iou_token_out, src, original_shape = (
            self.pretrained_segvol.mask_decoder.predict_mask_tokens(
                image_embeddings=image_embedding,
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
        )

        posterior_prototype = self.prototype_mlp(task_prior)

        posterior_prototype = posterior_prototype.view(1, 4, 768)
        mask_tokens_with_posterior = F.sigmoid(mask_tokens * posterior_prototype)

        # fmt: off
        low_res_masks_after_posterior, _ = self.pretrained_segvol.mask_decoder.predict_masks_from_tokens(
                mask_tokens_out=mask_tokens_with_posterior,
                iou_token_out=iou_token_out,
                src=src,
                original_shape=original_shape,
                text_embedding=text_embedding,
        )

        # fmt: on

        # Approach 2
        # posterior_prototype = posterior_prototype.view(1, 1, 32, 64, 64)

        # low_res_masks_expanded = low_res_masks.unsqueeze(
        #     1
        # )  # Shape: B, 1, C, 32, 64, 64
        # posterior_prototype_expanded = posterior_prototype.unsqueeze(
        #     0
        # )  # Shape: 1, T, C, 32, 64, 64

        # product = F.sigmoid(low_res_masks_expanded * posterior_prototype_expanded)

        # low_res_masks_after_posterior = product.sum(dim=2)  # Shape: B, T, 32, 64, 64

        # Paper approach (does it make sense?)

        # print(f"posterior_prototype.shape: {posterior_prototype.shape}")

        # low_res_masks_after_posterior = F.sigmoid(
        #     torch.einsum("btc,bcdhw->btdhw", posterior_prototype, low_res_masks)
        # )

        logits_per_task_after_posterior = F.interpolate(
            low_res_masks_after_posterior,
            size=img_shape,
            mode="trilinear",
            align_corners=False,
        )

        return logits_per_task_after_posterior


class SegVolContextPrior(SegVolModel):
    """
    SegVol model + using context priors as suggested in http://arxiv.org/abs/2103.00020
    """

    def __init__(
        self, config: SegVolConfig, context_prior_pool_config: ContextPriorPoolConfig
    ):
        super().__init__(config)

        self.model: SegVol = AutoModel.from_pretrained(
            "BAAI/SegVol", trust_remote_code=True, test_mode=config.test_mode
        ).model

        clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        self.model.pretrained_segvol.text_encoder.tokenizer = clip_tokenizer

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
            use_group_embeddings=self.config.use_group_embeddings,
        )
        # load the pre-trained weights
        pooled_prompt_encoder.load_from_pretrained(self.model.text_encoder.state_dict())
        # replace the original prompt encoder
        self.model.prompt_encoder = pooled_prompt_encoder

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
