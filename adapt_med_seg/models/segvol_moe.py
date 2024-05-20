import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from peft import PeftModel, LoraConfig, get_peft_model
from SegVol.model_segvol_single import SegVolConfig, SegVolProcessor
from adapt_med_seg.models.segvol_base import SegVolBase

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class ExpertModel(nn.Module):
    def __init__(self, model_name, r, alpha, dropout):
        super(ExpertModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name).model
        clip_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.text_encoder.tokenizer = clip_tokenizer
        
        peft_config = LoraConfig(
            target_modules=["out_proj", "qkv", "linear1", "linear2"],
            inference_mode=False,
            r=r,
            use_rslora=True,
            lora_alpha=alpha,
            lora_dropout=dropout,
        )
        self.model: PeftModel = get_peft_model(self.model, peft_config)

    def forward(self, x):
        return self.model(x)

class SegVolMoE(SegVolBase, PreTrainedModel):
    """
    SegVol model with Mixture of Experts.
    """

    def __init__(self, config: SegVolConfig, num_experts=2, r=8, alpha=8, dropout=0.0):
        super().__init__(config)
        self.num_experts = num_experts

        # Create expert models
        self.experts = nn.ModuleList([ExpertModel("BAAI/SegVol", r, alpha, dropout) for _ in range(num_experts)])
        
        # Create gating network
        self.gating_network = GatingNetwork(config.input_dim, num_experts)
        
        # Initialize the processor
        self.processor = SegVolProcessor(spatial_size=self.config.spatial_size)

    def forward(self, image, zoomed_image=None, text_prompt=None, modality=None):
        # Prepare input for gating network
        gating_input = self.prepare_gating_input(image, text_prompt)
        gating_weights = self.gating_network(gating_input)

        # Forward pass through each expert
        expert_outputs = [expert(image) for expert in self.experts]

        # Combine expert outputs based on gating weights
        combined_output = self.combine_expert_outputs(expert_outputs, gating_weights)
        return combined_output

    def prepare_gating_input(self, image, text_prompt):
        # Example logic to prepare input for gating network
        # Here we just flatten the image and concatenate with text features
        if text_prompt is not None:
            text_features = self.experts[0].model.text_encoder(text_prompt)["pooler_output"]
            image_features = image.view(image.size(0), -1)  # Flatten the image
            gating_input = torch.cat((image_features, text_features), dim=1)
        else:
            gating_input = image.view(image.size(0), -1)
        return gating_input

    def combine_expert_outputs(self, expert_outputs, gating_weights):
        combined_output = sum(weight * output for weight, output in zip(gating_weights, expert_outputs))
        return combined_output
    
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

    def save_pretrained(self, path: str):
        for idx, expert in enumerate(self.experts):
            expert.model.save_pretrained(f"{path}/expert_{idx}")
        # Save the gating network separately if needed

    def train(self, mode: bool=True):
        """
        Set the model to training mode.
        For CT, adapters are disabled. For MRI, LoRA adapters are trained.
        """
        self.training = mode
        for expert in self.experts:
            expert.train(mode)
        self.gating_network.train(mode)

    def disable_adapters(self):
        """Context manager to temporarily disable LoRA adapters."""
        class DisableAdapters:
            def __enter__(inner_self):
                for expert in self.experts:
                    expert.model.disable_adapters()
            def __exit__(inner_self, exc_type, exc_val, exc_tb):
                for expert in self.experts:
                    expert.model.enable_adapters()
        return DisableAdapters()
