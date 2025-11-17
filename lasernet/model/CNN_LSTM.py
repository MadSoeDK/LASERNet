
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torchvision import transforms

#path_to_file = "/Users/eva/Documents/DataSpatiotemporal/00/00_t_1.tiff"

#img = Image.open(path_to_file)
#to_tensor = transforms.ToTensor()
#img_tensor = to_tensor(img)

#print("Tensor shape: ", img_tensor.shape)

class temp_CNN(torch.nn.Module):
    def __init__(self):
        super(temp_CNN, self).__init__()
        #in-size is [1, 1555, 2916] given that we flatten it first (greyscale)
        #padding 1 to keep size
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2), 

                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                 nn.ReLU(), 
                                 nn.Conv2d(64, 128, 3, padding=1), 
                                 nn.ReLU(), 
                                 nn.MaxPool2d(2), 
                                 
                                 #refine with 128 --> 128 before pooling again
                                 nn.Conv2d(128, 128, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 128, 3, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2),

                                 nn.Conv2d(128, 256, 3, padding=1),
                                 nn.ReLU(), 
                                 nn.MaxPool2d(2)) #each maxpooling reduces the size by half                 
    def forward(self, x):
        return self.cnn(x)
    
class CNN_LSTM(torch.nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        self.cnn = temp_CNN()

        #change according to how much ram we have
        self.pooled_H = 4
        self.pooled_W = 4
        self.spatial_pool = nn.AdaptiveAvgPool2d((self.pooled_H, self.pooled_W))
        
        self.flatten = nn.Flatten()

        feature_dim = 256 * self.pooled_H * self.pooled_W
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=512, num_layers=2, batch_first=True) #512 enternal feature size

        #self.fc = nn.Linear(512, 1) #just for debuggning, output a single value
        self.fc = nn.Linear(512, feature_dim) 
    
    def encode_frame(self, frame):
        f = self.cnn(frame)                     # [batch, 256, 97, 182]
        f = self.spatial_pool(f)                # [batch, 256, 4, 4]
        f = self.flatten(f)                     # [batch, 4096]
        return f
    
    def forward(self, seq):
        #seq = [batch, seq_len, channels, height, width]
        batch_size, seq_len, C, H, W = seq.size()
        cnn_features = []
        for t in range(seq_len):
            f = self.cnn(seq[:, t]) #one frame through the cnn
            f = self.spatial_pool(f) #[batch, 256, 4, 4]
            f = self.flatten(f) #flatten to vector [batch, 4096]
            cnn_features.append(f)
        
        #stack the frames across time
        cnn_features = torch.stack(cnn_features, dim=1) #[batch, seq_len, feature_dim]

        #LSTM over time dimension
        lstm_out, _ = self.lstm(cnn_features)

        #output from last time step 
        out = self.fc(lstm_out[:, -1]) #[batch, feature_dim]
        out = out.view(batch_size, 256, self.pooled_H,self.pooled_W)

        return out 

#model = CNN_LSTM()

#fake_seq = torch.randn(2, 3, 1, 1555, 2916)  # batch=2, 3 frames
#out = model(fake_seq)

#print(out.shape)

#x = torch.randn((1, 1, 1555, 2916)) #batch size of 1, greyscale
#model = temp_CNN()
#y = model(x)
#print(y.shape)
#shape is now [1, 256, 97, 182] after 4 maxpoolings

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # 32→64
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


