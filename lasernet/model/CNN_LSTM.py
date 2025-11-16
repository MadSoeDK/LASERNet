
import torch
import torch.nn as nn

class TempEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [B, 1, 187, 929]
        self.enc1 = self._conv_block(1, 32)      # [B, 32, 187, 929]
        self.enc2 = self._conv_block(32, 64)     # [B, 64, 93, 464] after pool
        self.enc3 = self._conv_block(64, 128)    # [B, 128, 46, 232] after pool
        self.enc4 = self._conv_block(128, 256)   # [B, 256, 23, 116] after pool
        self.enc5 = self._conv_block(256, 512)   # [B, 512, 11, 58] after pool

        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),  # Add BN for stability
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.pool(e1)
        e2 = self.enc2(e2)
        e3 = self.pool(e2)
        e3 = self.enc3(e3)
        e4 = self.pool(e3)
        e4 = self.enc4(e4)
        e5 = self.pool(e4)
        e5 = self.enc5(e5)
        return e5, [e1, e2, e3, e4]  # Return skip connections for decoder

class TempDecoder(nn.Module):
    """Upsample with skip connections (U-Net style)"""
    def __init__(self):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._conv_block(512, 256)  # 256 from up + 256 from skip

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)  # 128 from up + 128 from skip

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block(64, 32)

        self.final = nn.Conv2d(32, 1, 1)
    
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    
    def forward(self, x, skips):
        e1, e2, e3, e4 = skips

        # Upsample and match e4 size exactly
        x = self.up4(x)
        x = nn.functional.interpolate(x, size=e4.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e4], dim=1)
        x = self.dec4(x)

        # Upsample and match e3 size exactly
        x = self.up3(x)
        x = nn.functional.interpolate(x, size=e3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)

        # Upsample and match e2 size exactly
        x = self.up2(x)
        x = nn.functional.interpolate(x, size=e2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        # Upsample and match e1 size exactly
        x = self.up1(x)
        x = nn.functional.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        x = self.final(x)
        # Crop/pad to exact [187, 929]
        return nn.functional.interpolate(x, size=(187, 929), mode='bilinear', align_corners=False)

class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TempEncoder()
        self.decoder = TempDecoder()

        # Work with [B, 512, 11, 58] features after 5 pooling layers
        feature_dim = 512 * 11 * 58  # 326,656
        self.flatten = nn.Flatten()

        # Project down to avoid cuDNN limitations (327k -> 1024)
        lstm_input_dim = 1024
        self.pre_lstm_proj = nn.Linear(feature_dim, lstm_input_dim)

        # LSTM on projected features
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=1024, num_layers=2, batch_first=True)

        # Project back to spatial
        self.fc = nn.Linear(1024, feature_dim)
    
    def forward(self, seq):
        """
        seq: [batch, seq_len, 1, 187, 929]
        returns: [batch, 1, 187, 929]
        """
        batch_size, seq_len = seq.size(0), seq.size(1)

        # Encode sequence
        features = []
        skip_connections = None
        for t in range(seq_len):
            feat, skips = self.encoder(seq[:, t])  # [B, 512, 11, 58]
            flat_feat = self.flatten(feat)  # [B, 326656]
            proj_feat = self.pre_lstm_proj(flat_feat)  # [B, 1024]
            features.append(proj_feat)
            if t == seq_len - 1:  # Keep last frame's skips for decoder
                skip_connections = skips

        features = torch.stack(features, dim=1)  # [B, seq_len, 1024]

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1]  # [B, 1024]

        # Reshape to spatial
        spatial_feat = self.fc(last_hidden)  # [B, 326656]
        spatial_feat = spatial_feat.view(batch_size, 512, 11, 58)

        # Decode with skip connections
        pred = self.decoder(spatial_feat, skip_connections)
        return pred

