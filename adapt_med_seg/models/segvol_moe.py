from SegVol.model_segvol_single import SegVolConfig, SegVolProcessor

from adapt_med_seg.models.segvol_base import SegVolBase
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

from peft import PeftModel, LoraConfig, get_peft_model

class SegVolMoE(SegVolBase, PreTrainedModel):
    """
    Baseline model for SegVol.
    """

    def __init__(self, config: SegVolConfig, r=8, alpha=8, dropout=0.0):
        super().__init__(config)

        self.model = AutoModel.from_pretrained(
            "BAAI/SegVol", trust_remote_code=True, test_mode=config.test_mode
        ).model

        clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        self.model.text_encoder.tokenizer = clip_tokenizer

        self.processor = SegVolProcessor(spatial_size=self.config.spatial_size)

        peft_config = LoraConfig(
            target_modules=["out_proj", "qkv", "linear1", "linear2"], # all ViT Linear layers
            inference_mode=config.test_mode,
            r=r,
            use_rslora=True,
            lora_alpha=alpha,
            lora_dropout=dropout,
        )

        self.model: PeftModel = get_peft_model(self.model, peft_config)

    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)

    def forward_test(self,
                     image,
                     zoomed_image=None,
                     text_prompt=None,
                     bbox_prompt_group=None,
                     point_prompt_group=None,
                     use_zoom=True,
                     modality=None):
        if modality != '1' and modality != 1 and modality != 'MRI':
            # CT mode
            with self.disable_adapters():
                return super().forward_test(image, zoomed_image, text_prompt, bbox_prompt_group, point_prompt_group, use_zoom, modality)
        else:
            # MRI mode
            return super().forward_test(image, zoomed_image, text_prompt, bbox_prompt_group, point_prompt_group, use_zoom, modality)

    def forward_train(self, image, train_organs, train_labels, modality):
        if modality != '1' and modality != 1 and modality != 'MRI':
            # CT mode
            with self.disable_adapters():
                return super().forward_train(image, train_organs, train_labels, modality)
        else:
            # MRI mode
            return super().forward_train(image, train_organs, train_labels, modality)

    def train(self, mode: bool=True):
        """
            Train MoE model for a specific modality.
            Currently only CT and MRI are supported,
            for CT, we throw an error because we don't want to train the pretrained model
            for MRI, we train the LoRA adapters.
        """
        self.train = True
        self.model.model.train(mode)
        self.model.model.test_mode = not mode
        self.config.test_mode = not mode
