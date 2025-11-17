!IMPORTANT!
The model outputs a downsized vector in shape [batch_size, 256, 4, 4]
The target values need to be changed for initial training to work: 
Like this: 
pred = model(seq)
target_latent = model.encode_frame(target)
loss = criterion(pred, target_latent)

Architecture: 

CNN input: [batch, 1, H, W]
CNN output: [batch, 256, 97, 182]   # after 4× downscaling via max-pooling
Pooling gives [batch, 256, 4, 4]
feature_dim = 256 × 4 × 4 = 4096

LSTM input: [batch, seq_len, feature_dim]
LSTM output: Last time step → [batch, 512]
[batch, 4096] → reshaped to [batch, 256, 4, 4]

Things to play around with with the CNN: 
- Number of convolution layers (currently 6)
- Number of max-pools (currently 4 → 16× downscaling)
- Refinement blocks (same-channel convs like 128→128)
- Adaptive pooled size (now 4×4, can increase for more detail)

Things to play around with with the LSTM: 
-hidden_size (currently 512, larger = more temporal capacity)
- num_layers (currently 2)
- dropout (currently 0, can add 0.1–0.3 for regularization)

Training parameters: 
- default 1e-4
- batch size (1-2 because of large images)
- Loss: MSE between predicted latent and target latent (or later, full-res MSE if we add a decoder)