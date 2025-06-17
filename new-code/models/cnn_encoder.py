import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # (B, T, 3)
        print("X.shape before encoder:", x.shape)
        if x.dim() == 1:
            x = x.view(1, 3, -1)  # Adjust this based on your data

        x = x.permute(0, 2, 1)   # (B, 3, T)
        x = self.encoder(x).squeeze(-1)  # (B, H)
        return self.linear(x)  # (B, output_dim)
