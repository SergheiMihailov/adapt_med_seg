from typing_extensions import Self
from transformers import AutoTokenizer, AutoModel
from SegVol.model_segvol_single import (
    SegVolConfig,
    SegVolModel,
    SegVolProcessor,
)

from peft import LoraConfig, get_peft_model

class SegVolBase(SegVolModel):
    """
    Baseline model for SegVol.
    """

    def __init__(self, config: SegVolConfig):
        super().__init__(config)

        self.model = AutoModel.from_pretrained(
            "BAAI/SegVol", trust_remote_code=True, test_mode=config.test_mode
        ).model

        clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        self.model.text_encoder.tokenizer = clip_tokenizer

        self.processor = SegVolProcessor(spatial_size=self.config.spatial_size)

    def eval(self) -> Self:
        self.model.eval()

    def train(self, mode = True) -> Self:
        self.model.train(mode)

    def to_cpu(self) -> Self:
        self.model.to("cpu")

    def to_cuda(self) -> Self:
        self.model.to("cuda")

    def to_mps(self) -> Self:
        self.model.to("mps")
        
        
class SegVolLoraViT(SegVolBase):
    def __init__(self, config: SegVolConfig, lora_config: LoraConfig):
        super().__init__(config)
        self.model = get_peft_model(self.model, lora_config)

class SegVolLoraAll(SegVolBase):
    def __init__(self, config: SegVolConfig, lora_config: LoraConfig):
        super().__init__(config)
        self.model = get_peft_model(self.model, lora_config)