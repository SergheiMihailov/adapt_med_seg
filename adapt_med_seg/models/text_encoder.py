from collections import defaultdict
from torch import nn
from transformers import CLIPTextModel, CLIPTextConfig

MOD_TO_PROMPT = defaultdict(
    {
        "CT": "A computerized tomography (CT) scan of a {}.",
        "MRI": "A magnetic resonance imaging (MRI) scan of a {}.",
        "PET": "A positron emission tomography (PET) scan of a {}.",
        "US": "An ultrasound (US) scan of a {}.",
    },
    default="A scan of a {}.",
)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        config = CLIPTextConfig()
        self.clip_text_model = CLIPTextModel(config)
        self.tokenizer = None
        self.dim_align = nn.Linear(512, 768)
        # freeze text encoder
        for param in self.clip_text_model.parameters():
            param.requires_grad = False

    def organ2tokens(self, organ_names: list[str], modality: str, device=None):
        print("__in__organ2tokens")
        text_list = [
            MOD_TO_PROMPT[modality].format(organ_name) for organ_name in organ_names
        ]
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt")
        for key in tokens.keys():
            tokens[key] = tokens[key].to(device)
        return tokens

    def forward(self, text, modality, device):
        if text is None:
            return None
        if isinstance(text, str):
            # text is supposed to be list
            text = [text]
        tokens = self.organ2tokens(text, modality, device)
        clip_outputs = self.clip_text_model(**tokens)
        text_embedding = clip_outputs.pooler_output
        text_embedding = self.dim_align(text_embedding)
        return text_embedding
