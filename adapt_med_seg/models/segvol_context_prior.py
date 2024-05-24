from ast import List
import logging
from typing import Optional, Tuple, Type

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel, AutoTokenizer
from typing_extensions import Self

from SegVol.model_segvol_single import (
    SegVol,
    SegVolConfig,
    SegVolModel,
    SegVolProcessor,
)

import torch.nn.functional as F

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
        feature_dim: int,  # 32*64*64 for low_res masks, 4*768 for mask tokens
        embed_dim: int = 512,
        task_prior_len: int = 1,
    ):
        super(PosteriorPrototypeMLP, self).__init__()
        self.embed_dim = embed_dim
        self.task_prior_len = task_prior_len

        reduction = 16  # TODO: remove reduction from here once we apply it to embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * self.task_prior_len, embed_dim // reduction),
            nn.ReLU(),
            nn.Linear(embed_dim // reduction, C * t_k * feature_dim),
        )

    def forward(
        self,
        task_prior: torch.Tensor,
    ):
        return self.mlp(task_prior)


class CustomMaskDecoder(nn.Module):
    def __init__(self, pretrained_segvol_mask_decoder):
        super(CustomMaskDecoder, self).__init__()
        self.mask_decoder = pretrained_segvol_mask_decoder

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embedding: Optional[torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_mask_tokens: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Returns:
          torch.Tensor: batched predicted masks
        """
        # print('--------------decoder here--------------')
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            text_embedding=text_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            output_mask_tokens=output_mask_tokens,
        )

        if output_mask_tokens:
            mask_tokens_out, iou_token_out = masks, iou_pred
            return mask_tokens_out, iou_token_out

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, mask_slice, :, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        text_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        output_mask_tokens: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.mask_decoder.iou_token.weight, self.mask_decoder.mask_tokens.weight],
            dim=0,
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w, d = src.shape

        # Run the transformer
        hs, src = self.mask_decoder.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.mask_decoder.num_mask_tokens), :]

        if output_mask_tokens:
            return mask_tokens_out, iou_token_out

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w, d)
        upscaled_embedding = self.mask_decoder.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.mask_decoder.num_mask_tokens):
            hyper_in_list.append(
                self.mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w, d = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w * d)).view(
            b, -1, h, w, d
        )

        if text_embedding is not None:
            text_embedding_down = self.mask_decoder.txt_align_upscaled_embedding(
                text_embedding
            ).unsqueeze(dim=1)
            upscaled_embedding = upscaled_embedding.view(b, c, h * w * d)
            sim = (text_embedding_down @ upscaled_embedding).view(b, -1, h, w, d)
            sim = sim.repeat(1, masks.shape[1], 1, 1, 1)
            masks = masks + sim
        iou_pred = self.mask_decoder.iou_prediction_head(iou_token_out)

        return masks, iou_pred


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
            feature_dim=1,
        )
        self.pretrained_segvol.mask_decoder = CustomMaskDecoder(
            pretrained_segvol_mask_decoder=self.pretrained_segvol.mask_decoder,
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
            logits = self.forward_decoder(
                image_embedding,
                img_shape,
                text,
                boxes,
                points,
                posterior_prototype=posterior_prototype,
                # modality=modality,
            )

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
            logits = self.forward_decoder(
                image_embedding,
                img_shape,
                text=organs,
                boxes=bboxes,
                points=points,
                posterior_prototype=posterior_prototype,
                # modality=modality,
            )

            # cal loss
            sl_loss_dice = self.pretrained_segvol.dice_loss.forward(
                logits.squeeze().float(), train_labels.squeeze().float()
            )
            sl_loss_bce = self.pretrained_segvol.bce_loss.forward(
                logits.squeeze().float(), train_labels.squeeze().float()
            )
            sl_loss += sl_loss_dice + sl_loss_bce
        return sl_loss

    def forward_decoder(
        self,
        image_embedding,
        img_shape,
        text=None,
        boxes=None,
        points=None,
        posterior_prototype=None,
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
        low_res_masks, _ = self.pretrained_segvol.mask_decoder(
            image_embeddings=image_embedding,
            text_embedding=text_embedding,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            output_mask_tokens=False,
        )

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

        print(f"posterior_prototype.shape: {posterior_prototype.shape}")

        low_res_masks_after_posterior = torch.einsum(
            "btc,bcdhw->btdhw", posterior_prototype, low_res_masks
        )

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
