from typing_extensions import Self
from transformers import AutoTokenizer, AutoModel
from SegVol.model_segvol_single import (
    SegVolConfig,
    SegVolModel,
    SegVolProcessor,
)

class SegVolContextPrior(SegVolModel):
    """
    SegVol model + using context priors as suggested in http://arxiv.org/abs/2103.00020
    """

    def __init__(self, config: SegVolConfig):
        super().__init__(config)

        self.model = AutoModel.from_pretrained(
            "BAAI/SegVol", trust_remote_code=True, test_mode=config.test_mode
        ).model

        print(f"keys: {self.model.keys()}")

        clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        self.model.text_encoder.tokenizer = clip_tokenizer

        self.processor = SegVolProcessor(spatial_size=self.config.spatial_size)

    def eval(self) -> Self:
        self.model.train(False)

    def train(self, mode = True) -> Self:
        self.model.train(mode)
        self.model.test_mode = not mode
        self.config.test_mode = not mode

    def to_cpu(self) -> Self:
        self.model.to("cpu")

    def to_cuda(self) -> Self:
        self.model.to("cuda")

    def to_mps(self) -> Self:
        self.model.to("mps")
        

