from models.cnn_encoder import CNNEncoder
from models.llama_regressor import LlamaRegressor
from utils import load_all_viewport_data
import torch.nn as nn, torch.optim as optim, torch

X, y = load_all_viewport_data("data/")

model = LlamaRegressor()
encoder = CNNEncoder()
optimizer = optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(10):
    model.train()
    print("X.shape before encoder:", X.shape)  # Add this before calling encoder

    emb = encoder(X)  # (B, 768)
    output = model(emb.unsqueeze(1))  # (B, 3)
    loss = loss_fn(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item():.4f}")
