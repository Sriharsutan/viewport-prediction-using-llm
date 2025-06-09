import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.linear = nn.Linear(128, out_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.encoder(x).squeeze(-1)  # (B, 128)
        return self.linear(x)  # (B, out_dim)

class ViewportPredictor(nn.Module):
    def __init__(self, llama_id="NousResearch/Llama-2-7b-hf"):
        super().__init__()
        self.encoder = CNNEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained(llama_id, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(llama_id, torch_dtype=torch.float32)

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            inference_mode=False
        )
        self.model = get_peft_model(self.model, config)

    def forward(self, inputs, labels=None):
        embedded = self.encoder(inputs)
        embedded = embedded.unsqueeze(1)
        return self.model(inputs_embeds=embedded, labels=labels)
