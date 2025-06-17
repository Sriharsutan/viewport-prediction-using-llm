import torch.nn as nn
import torch

class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear
        self.lora_A = nn.Parameter(torch.randn(base_linear.out_features, r))
        self.lora_B = nn.Parameter(torch.randn(r, base_linear.in_features))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.lora_B.T @ self.lora_A.T)

def inject_lora(model, r=8, alpha=16):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            setattr(model, name.split('.')[-1], LoRALinear(module, r, alpha))
