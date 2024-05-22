from typing_extensions import Self
from transformers import AutoTokenizer, AutoModel
from SegVol.model_segvol_single import (
    SegVolConfig,
    SegVolModel,
    SegVolProcessor,
    TextEncoder,
)


class SegVolBase(SegVolModel):
    """
    Baseline model for SegVol.
    """

    def __init__(self, config: SegVolConfig):
        super().__init__(config)

        self.model = AutoModel.from_pretrained(
            "BAAI/SegVol", trust_remote_code=True, test_mode=config.test_mode
        ).model

        # Overwrite the text encoder
        self.model.text_encoder = TextEncoder()
        # Load the tokenizer
        clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        self.model.text_encoder.tokenizer = clip_tokenizer

        self.processor = SegVolProcessor(spatial_size=self.config.spatial_size)

    def eval(self) -> Self:
        return self.model.train(False)

    def train(self, mode=True) -> Self:
        self.model.train(mode)
        self.model.test_mode = not mode
        self.config.test_mode = not mode
        return self

    def to_cpu(self) -> Self:
        self.model.to("cpu")
        return self

    def to_cuda(self) -> Self:
        self.model.to("cuda")
        return self

    def to_mps(self) -> Self:
        self.model.to("mps")
        return self

    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)
