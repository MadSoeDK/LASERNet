import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
print("PYTHONPATH added:", ROOT)

from lasernet.model.CNN_LSTM import CNN_LSTM
from src.dataset_images.dataloader import get_image_dataloaders

import torch
import torch.nn as nn
import torch.optim as optim

root_dir = "/zhome/ef/5/219124/LASERNet/DataSpatiotemporal/"

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU only")

#Dataloader
train_loader, val_loader, test_loader = get_image_dataloaders(
    root_dir=root_dir,
    seq_length=5,
    batch_size=1)


# Model
model = CNN_LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for seq, target in train_loader:
        seq = seq.float().to(device) #[B, seq, 1, H, W]
        target = target.float().to(device) #[B, 1, H, W]

        optimizer.zero_grad()

        pred = model(seq)                        # [batch, 4096]
        pred = pred.view(pred.size(0), -1)       # [B, 4096]
        target_latent = model.encode_frame(target)  # [batch, 4096]

        loss = criterion(pred, target_latent)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch}, train loss = {train_loss/len(train_loader)}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for seq, target in val_loader:
            seq = seq.float().to(device)
            target = target.float().to(device)

            pred = model(seq)
            pred = pred.view(pred.size(0), -1)       # [B, 4096]
            target_latent = model.encode_frame(target)

            loss = criterion(pred, target_latent)
            val_loss += loss.item()

    print(f"         val loss = {val_loss/len(val_loader)}")

