import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import v2
import matplotlib.pyplot as plt


## Helper Functions
def MSE(x, x_hat):
    mse_val = torch.sum((x - x_hat)**2)
    return mse_val 

## Custom Data Loader
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image

my_transforms = v2.Compose([v2.ToTensor(),
                            v2.RandomResizedCrop(size=(256, 256), antialias=True),
                            v2.RandomHorizontalFlip(p=0.20),
                            v2.ToDtype(torch.float32, scale=True)
                           ])

dataset_path = "/Users/hannah_neumann/Downloads/train/"           

training_data = CustomImageDataset(img_dir=dataset_path,
                                   transform=my_transforms)


train_dataloader = DataLoader(training_data, batch_size=1)


## My Neural Network Models
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,   out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=12,  kernel_size=5, stride=1, padding=2)
        
        self.act1 = nn.ReLU(inplace=False)
        self.act2 = nn.ReLU(inplace=False)
        self.act3 = nn.ReLU(inplace=False)
        self.act4 = nn.ReLU(inplace=False)
        self.act5 = nn.ReLU(inplace=False)
        self.act6 = nn.Tanh()  

    def forward(self, x):
        layer1 = self.act1(self.conv1(x))       # [B,   3, 256, 256] --> [B, 128, 256, 256]

        layer2 = self.act2(self.conv2(layer1))  # [B, 128, 256, 256] --> [B, 256, 128, 128], downsample!
        layer3 = self.act3(self.conv3(layer2))  # [B, 256, 128, 128] --> [B, 256, 128, 128]

        layer4 = self.act4(self.conv4(layer3))  # [B, 256, 128, 128] --> [B, 512,  64,  64], downsample!
        layer5 = self.act5(self.conv5(layer4))  # [B, 512,  64,  64] --> [B, 512,  64,  64]

        layer6 = self.act6(self.conv6(layer5))  # [B, 512,  64,  64] --> [B,  12,  64,  64]
        y      = 32.0 * layer6                  # brings data from [-1,+1] to [-32, +32]

        return y

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=12,  out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)

        self.conv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128,  kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=128, out_channels=3,  kernel_size=3, stride=1, padding=1)
        
        self.act1 = nn.ReLU(inplace=False)
        self.act2 = nn.ReLU(inplace=False)
        self.act3 = nn.ReLU(inplace=False)
        self.act4 = nn.ReLU(inplace=False)
        self.act5 = nn.ReLU(inplace=False)
        self.act6 = nn.ReLU(inplace=False)
        self.act7 = nn.Sigmoid()  

    def forward(self, y):
        layer1 = self.act1(self.conv1(y))       # [B,  12,  64,  64] --> [B, 512,  64,  64]
        layer2 = self.act2(self.conv2(layer1))  # [B, 512,  64,  64] --> [B, 512,  64,  64]

        layer3 = self.act3(self.conv3(layer2))  # [B, 512,  64,  64] --> [B, 256, 128, 128], upsample
        layer4 = self.act4(self.conv4(layer3))  # [B, 256, 128, 128] --> [B, 256, 128, 128]

        layer5 = self.act5(self.conv5(layer4))  # [B, 256, 128, 128] --> [B, 128, 256, 256]
        layer6 = self.act6(self.conv6(layer5))  # [B, 128, 256, 256] --> [B, 128, 256, 256]
        layer7 = self.act7(self.conv7(layer6))  # [B, 128, 256, 256] --> [B,   3, 256, 256]
        x_hat  = layer7

        return x_hat


## Define Everything
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
learning_rate = 1e-4

compression_encoder = Encoder().to(device)
compression_decoder = Decoder().to(device)

optim = torch.optim.Adam([*compression_encoder.parameters(), *compression_decoder.parameters()],
                          lr=learning_rate)

## Training Loop
for counter, input_img in enumerate(train_dataloader):
    optim.zero_grad()

    x = input_img.to(device)        # [B,  3, 256, 256]
    y = compression_encoder(x)      # [B, 12,  64,  64]
    x_hat = compression_decoder(y)  # [B,  3, 256, 256]

    distortion_loss = MSE(x, x_hat)
    loss = distortion_loss

    loss.backward()
    optim.step()

    print(counter, loss)
