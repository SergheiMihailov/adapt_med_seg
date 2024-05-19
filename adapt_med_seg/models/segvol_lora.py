from SegVol.model_segvol_single import SegVolConfig, SegVolModel, SegVolProcessor

from adapt_med_seg.models.segvol_base import SegVolBase
from transformers import AutoModel, AutoTokenizer

from peft import PeftModel, LoraConfig, get_peft_model

class SegVolLORA(SegVolModel):
    """
    Baseline model for SegVol.
    """

    def __init__(self, config: SegVolConfig, r=8, alpha = 8, dropout = 0.0):
        super().__init__(config)

        self.model = AutoModel.from_pretrained(
            "BAAI/SegVol", trust_remote_code=True, test_mode=config.test_mode
        ).model

        clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        self.model.text_encoder.tokenizer = clip_tokenizer

        self.processor = SegVolProcessor(spatial_size=self.config.spatial_size)

        peft_config = LoraConfig(
            target_modules=["q_proj", "v_proj"],
            inference_mode=config.test_mode,
            r=r,
            use_rslora=True,
            lora_alpha=alpha,
            lora_dropout=dropout,
        )

        self.model = PeftModel(self.model, peft_config)
        
    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)

    def train(self, mode = True):
        self.model.model.train(mode)
        self.model.model.test_mode = not mode
        self.config.test_mode = not mode

class SegVolLoraViT(SegVolBase):
    def __init__(self, config: SegVolConfig):
        super().__init__(config)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=["decode_head"]
        )
        self.model = get_peft_model(self.model, lora_config)

class SegVolLoraAll(SegVolBase):
    def __init__(self, config: SegVolConfig):
        super().__init__(config)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, lora_config)