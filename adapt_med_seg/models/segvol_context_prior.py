import logging
import re
from dataclasses import dataclass
from typing import Mapping, Tuple, Type

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer
from typing_extensions import Self

from SegVol.model_segvol_single import PromptEncoder as SamPromptEncoder
from SegVol.model_segvol_single import (
    SegVol,
    SegVolConfig,
    SegVolModel,
    SegVolProcessor,
)

REDUCTION = 16

logger = logging.getLogger(__name__)


class ContextPriorPool(nn.Module):
    def __init__(
        self,
        tasks: list[str],
        modalities: list[str],
        embed_dim: int = 768,
        modality_prior_len: int = 1,
        task_prior_len: int = 1,
        dtype: Type[torch.dtype] = torch.float32,
    ):
        super(ContextPriorPool, self).__init__()

        self.tasks = tasks
        self.modalities = modalities
        self.embed_dim = embed_dim

        self.task_prior_embeddings = nn.ParameterDict(
            {
                task: nn.Parameter(torch.zeros(task_prior_len, embed_dim, dtype=dtype))
                for task in self.tasks
            }
        )

        self.modality_prior_embeddings = nn.ParameterDict(
            {
                modality: nn.Parameter(
                    torch.zeros(modality_prior_len, embed_dim, dtype=dtype)
                )
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


class PriorFusion(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 4):
        super(PriorFusion, self).__init__()
        reduction = REDUCTION

        self.embed_dim = embed_dim

        self.group_embeddings = nn.Parameter(
            torch.zeros(3, embed_dim, requires_grad=True)
        )
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction),
            nn.ReLU(),
            nn.Linear(embed_dim // reduction, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        image_rep: torch.Tensor,
        modality_prior: torch.Tensor,
        task_prior: torch.Tensor,
    ):
        # MHA
        image_rep = image_rep.permute(1, 0, 2) + self.group_embeddings[0]
        modality_prior = modality_prior + self.group_embeddings[1]
        task_prior = task_prior + self.group_embeddings[2]

        modality_prior_expanded = modality_prior.unsqueeze(0)  # Now [1, 1, 768]
        task_prior_expanded = task_prior.unsqueeze(0)  # Now [1, 1, 768]

        input_seq = torch.cat(
            [image_rep, modality_prior_expanded, task_prior_expanded], dim=0
        )

        output_seq, _ = self.mha(input_seq, input_seq, input_seq)

        # LayerNorm
        output_seq = self.norm1(output_seq)

        # FFN
        output_seq = self.ffn(output_seq)

        # LayerNorm
        output_seq = self.norm2(output_seq)

        output_seq = output_seq.permute(1, 0, 2)

        image_seq_len = image_rep.shape[0]
        modality_seq_len = modality_prior.shape[0]

        image_rep = output_seq[:, :image_seq_len, :]
        modality_prior = output_seq[
            :, image_seq_len : image_seq_len + modality_seq_len, :
        ]
        task_prior = output_seq[:, image_seq_len + modality_seq_len :, :]

        return image_rep, modality_prior, task_prior


class PosteriorPrototypeMLP(nn.Module):
    def __init__(
        self,
        t_k: int,
        C: int,
        embed_dim: int = 512,
        task_prior_len: int = 1,
    ):
        super(PosteriorPrototypeMLP, self).__init__()
        self.embed_dim = embed_dim
        self.task_prior_len = task_prior_len

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * self.task_prior_len, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, C * t_k),
        )

    def forward(
        self,
        task_prior: torch.Tensor,
    ):

        return self.mlp(task_prior)


class SegVolContextPriorModel(nn.Module):
    def __init__(
        self,
        pretrained_segvol: SegVolModel,
        config: SegVolConfig,
        test_mode: bool = False,
        tasks: list[str] = ["prostate"],
        modalities: list[str] = ["MRI"],
        embed_dim: int = 768,
    ):
        super(SegVolContextPriorModel, self).__init__()

        embed_dim = 768

        self.pretrained_segvol = pretrained_segvol
        self.config = config
        self.test_mode = test_mode

        # Context prior:
        self.context_prior_pool = ContextPriorPool(
            dtype=torch.float32,
            tasks=tasks,
            modalities=modalities,
            embed_dim=embed_dim,
        )

        self.prior_fusion = PriorFusion(embed_dim=embed_dim)
        self.prototype_mlp = PosteriorPrototypeMLP(
            t_k=1,  # TODO: Make this a parameter, and should be times number of tasks
            C=1,
            embed_dim=embed_dim,
            task_prior_len=1,
        )

        target_modules = ["query", "value", "key", "linear1", "linear2"]
        peft_config = LoraConfig(
            target_modules=target_modules,
            inference_mode=self.config.test_mode,
            use_rslora=True,
            r=8,
            lora_alpha=8,
            lora_dropout=0.0,
        )

        self.pretrained_segvol = get_peft_model(self.pretrained_segvol, peft_config)

    def forward_train(self, image, train_organs, train_labels, modality):
        loss = self.model(
            image,
            text=None,
            boxes=None,
            points=None,
            train_organs=train_organs,
            train_labels=train_labels,
            modality=modality,
        )
        return loss

    def forward(
        self,
        image,
        text=None,
        boxes=None,
        points=None,
        modality: str = "MRI",
        task: str = "prostate",
        **kwargs,
    ):

        modality_prior = self.context_prior_pool.get_modality_prior(modality)
        task_prior = self.context_prior_pool.get_task_prior(task)

        bs = image.shape[0]
        img_shape = (image.shape[2], image.shape[3], image.shape[4])
        image_embedding, _ = self.pretrained_segvol.image_encoder(image)

        image_embedding, modality_prior, task_prior = self.prior_fusion(
            image_embedding, modality_prior, task_prior
        )

        posterior_prototype = self.prototype_mlp(task_prior)

        image_embedding = image_embedding.transpose(1, 2).view(
            bs,
            -1,
            int(self.pretrained_segvol.feat_shape[0]),
            int(self.pretrained_segvol.feat_shape[1]),
            int(self.pretrained_segvol.feat_shape[2]),
        )

        # test mode
        if self.test_mode:
            logits = self.pretrained_segvol.forward_decoder(
                image_embedding,
                img_shape,
                text,
                boxes,
                points,
                # modality=modality,
            )

            logits = torch.einsum("btc,bcdhw->btdhw", posterior_prototype, logits)

            return logits

        # train mode
        ## sl
        sl_loss = self.supervised_forward(
            image,
            image_embedding,
            img_shape,
            kwargs["train_organs"],
            kwargs["train_labels"],
            modality=modality,
            posterior_prototype=posterior_prototype,
        )
        ## ssl
        # ssl_loss = self.unsupervised_forward(image, image_embedding, kwargs['pseudo_seg_cleaned'], img_shape)
        return sl_loss

    def supervised_forward(
        self,
        image,
        image_embedding,
        img_shape,
        training_organs,
        train_labels,
        modality=None,
        posterior_prototype=None,
    ):
        device = image_embedding.device
        iter_points, iter_bboxes, iter_organs = (
            self.pretrained_segvol.build_prompt_label(
                image.shape[0], training_organs, train_labels, device
            )
        )
        # select prompt
        prompt_options = [
            [None, iter_points, iter_organs],
            [iter_bboxes, None, iter_organs],
            [None, None, iter_organs],
            [iter_bboxes, None, None],
            [None, iter_points, None],
            [iter_bboxes, iter_points, None],
        ]
        sl_loss = 0
        for prompt in prompt_options:
            bboxes, points, organs = prompt
            logits = self.pretrained_segvol.forward_decoder(
                image_embedding,
                img_shape,
                text=organs,
                boxes=bboxes,
                points=points,
                # modality=modality,
            )

            logits = torch.einsum("btc,bcdhw->btdhw", posterior_prototype, logits)

            # cal loss
            sl_loss_dice = self.pretrained_segvol.dice_loss.forward(
                logits.squeeze().float(), train_labels.squeeze().float()
            )
            sl_loss_bce = self.pretrained_segvol.bce_loss.forward(
                logits.squeeze().float(), train_labels.squeeze().float()
            )
            sl_loss += sl_loss_dice + sl_loss_bce
        return sl_loss


class SegVolContextPrior(SegVolModel):
    """
    SegVol model + using context priors as suggested in http://arxiv.org/abs/2103.00020
    """

    def __init__(self, config: SegVolConfig, **kwargs):
        super().__init__(config)

        pretrained_segvol = AutoModel.from_pretrained(
            "BAAI/SegVol", trust_remote_code=True, test_mode=config.test_mode
        ).model

        self.model: SegVol = SegVolContextPriorModel(
            pretrained_segvol=pretrained_segvol,
            config=config,
        )

        clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        self.model.pretrained_segvol.text_encoder.tokenizer = clip_tokenizer

        self.processor = SegVolProcessor(spatial_size=self.config.spatial_size)

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
