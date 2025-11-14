from model.CNN_LSTM import CNN_LSTM
from dataset.dataloader import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim

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

# Model
model = CNN_LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Data
train_loader, val_loader = get_dataloaders(batch_size=2, sequence_length=5)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for seq, target in train_loader:
        seq = seq.float().to(device)
        target = target.float().to(device)

        optimizer.zero_grad()

        pred = model(seq)                        # [batch, 4096]
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
            target_latent = model.encode_frame(target)

            loss = criterion(pred, target_latent)
            val_loss += loss.item()

    print(f"         val loss = {val_loss/len(val_loader)}")

