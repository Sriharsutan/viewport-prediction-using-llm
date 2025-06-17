from transformers import LlamaModel, LlamaConfig
from models.lora_adapter import inject_lora
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaRegressor(nn.Module):
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(model_id)
        self.proj = nn.Linear(768, 2048)  # Map CNN output dim → TinyLlama hidden size

        # Regression head: Take last hidden state, map to 3D (roll, pitch, yaw)
        self.reg_head = nn.Linear(2048, 3)

    def forward(self, embeddings):
        """
        embeddings: [B, 7, 768] ← from your CNN encoder
        """
        embeddings = self.proj(embeddings)  # Now shape is [B, 7, 2048]
        outputs = self.backbone(inputs_embeds=embeddings, return_dict=True)

        # Get final hidden state at the last token
        last_hidden = outputs.logits[:, -1, :]  # [B, vocab_size]

        # Option 1: Alternatively, use the last hidden state directly
        # hidden_states = outputs.hidden_states[-1][:, -1, :]  # [B, 2048]

        # Map to [B, 3] for regression
        predicted = self.reg_head(last_hidden)  # [B, 3]
        return predicted

