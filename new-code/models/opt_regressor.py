from transformers import OPTModel
from models.lora_adapter import inject_lora
import torch.nn as nn

class OPTRegressor(nn.Module):
    def __init__(self, model_name="facebook/opt-1.3b"):
        super().__init__()
        self.backbone = OPTModel.from_pretrained(model_name)
        inject_lora(self.backbone)
        self.head = nn.Linear(self.backbone.config.hidden_size, 3)

    def forward(self, embeddings):
        outputs = self.backbone(inputs_embeds=embeddings)
        return self.head(outputs.last_hidden_state[:, -1, :])
