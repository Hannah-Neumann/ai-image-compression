import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import v2
import matplotlib.pyplot as plt

import wandb
from tqdm import tqdm

## Helper Functions
def MSE(x, x_hat):
    mse_val = torch.sum((x - x_hat)**2)
    return mse_val 

# Rate Loss, Assumes a Normal Distribution (PMF)
def rate_loss_fun(y, mu=0, sigma=5):
    likelihood = get_likelihood(y, mu, sigma)
    filesize_in_bits = -torch.sum(torch.log2(likelihood))

    return filesize_in_bits

def get_likelihood(y, mu, sigma):
    y_noisy = y + (torch.randn_like(y) - 0.50)  # add uniform noise between [-0.5, 0.50]
    
    normal_dist = torch.distributions.normal.Normal(mu, sigma)
    likelihood = normal_dist.cdf(y_noisy + 0.50) - normal_dist.cdf(y_noisy - 0.50)

    return likelihood

class integer_quantization_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y):
        ctx.save_for_backward(y)
        y_hat = torch.round(y)  # round to integer
        return y_hat
    
    @staticmethod
    def backward(ctx, grad_output):
        local_gradient = 1  # STE
        return local_gradient*grad_output

## Custom Dataset Class
class TrainImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = read_image(img_path)
        
        # detecting grayscale images:
        if image.shape[0] == 1:
            image = torch.cat((image,image,image), dim=0)
        if self.transform:
            image = self.transform(image)
        return image

class ValidImageDataset(Dataset):
    def __init__(self, img_dir, img_divisible_by = 4, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        self.img_divisible_by = img_divisible_by

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = read_image(img_path)
        
        # ensuring the image height and width is divisible by 'self.img_divisible_by':
        new_height_dim = (image.shape[1] // self.img_divisible_by) * self.img_divisible_by
        new_width_dim = (image.shape[2] // self.img_divisible_by) * self.img_divisible_by
        image = image[:, 0:int(new_height_dim), 0:int(new_width_dim)]

        # detecting grayscale images:
        if image.shape[0] == 1:
            image = torch.cat((image,image,image), dim=0)
        if self.transform:
            image = self.transform(image)
        
        return image

my_transforms_training = v2.Compose([v2.ToTensor(),
                                     v2.RandomResizedCrop(size=(256, 256), antialias=True),
                                     v2.RandomHorizontalFlip(p=0.20),
                                     v2.ToDtype(torch.float32, scale=True)
                                    ])

my_transforms_validation = v2.Compose([v2.ToTensor(),
                                       v2.ToDtype(torch.float32, scale=True)
                                      ])

dataset_train_path = "datasets/train_CLIC2021/"   
dataset_validation_path = "datasets/valid_CLIC2021/"           

training_data = TrainImageDataset(img_dir=dataset_train_path,
                                  transform=my_transforms_training)

validation_data = ValidImageDataset(img_dir=dataset_validation_path,
                                    img_divisible_by = 4,
                                    transform=my_transforms_validation)


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

## Setting the Backend
use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available()

if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

train_kwargs = {'batch_size': 1, 'pin_memory': True, 'shuffle': True}
valid_kwargs = {'batch_size': 1, 'pin_memory': True, 'shuffle': False}

if use_cuda:
    cuda_train_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    cuda_valid_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': False}
    train_kwargs.update(cuda_train_kwargs)
    valid_kwargs.update(cuda_valid_kwargs)

print("\n\nDevice: ", device, "\n\n")
train_dataloader = DataLoader(training_data, **train_kwargs)
valid_dataloader = DataLoader(validation_data, **valid_kwargs)
                              
learning_rate = 1e-4

compression_encoder = Encoder().to(device)
compression_decoder = Decoder().to(device)

optim = torch.optim.Adam([*compression_encoder.parameters(), *compression_decoder.parameters()],
                          lr=learning_rate)

## WandB Tracking Initialization
wandb.init(
    # set the wandb project where this run will be logged
    project="AI-Image-Compression",
    name="test_2",
        
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.0001,
        "architecture": "AI Image Compression AutoEncoder",
        "dataset": "CLIC_2021",
        "epochs": 20,
        })
wandb.define_metric("*", step_metric="global_step")


## Validation Lopp:
def run_validation(valid_dataloader, global_step):
    val_counter = 0.0
    avg_distortion = 0.0
    avg_rate = 0.0
    avg_loss = 0.0

    for val_loop_counter, input_img in enumerate(tqdm(valid_dataloader)):
        with torch.no_grad():
            x = input_img.to(device)                     # [B,  3,     H,    W]
            y = compression_encoder(x)                   # [B, 12,  H//4, W//4]
            y_hat = integer_quantization_class.apply(y)  # quantize to integer; custom straight-through gradient
            x_hat = compression_decoder(y_hat)           # [B,  3,     H,    W]
            
            distortion_loss = MSE(x, x_hat)
            rate_loss = rate_loss_fun(y)
            loss = distortion_loss + 0.1 * rate_loss
            
            avg_distortion += distortion_loss
            avg_rate += rate_loss
            avg_loss += loss
            val_counter +=1
            
            # Log all Validation images:
            x_and_xhat = torch.cat((x, x_hat), dim=3).permute(0,2,3,1)  # [B,C,H,W] -> [B,H,W,C]
            x_and_xhat = x_and_xhat[0,:]                                # [B,H,W,C] -> [H,W,C]
            x_and_xhat = x_and_xhat.cpu().detach().numpy()              # send to cpu & numpy
            image = wandb.Image(x_and_xhat, caption=f"original and compressed image, MSE: {distortion_loss}")
            wandb.log({"global_step": global_step, f"valid/images_{val_loop_counter}": image})
    
    # Log aggrregated validation metrics:
    avg_distortion = avg_distortion / val_counter
    avg_rate = avg_rate / val_counter
    avg_loss = avg_loss / val_counter
    wandb.log({"global_step": global_step, "valid/MSE": 255.0*255.0*avg_distortion}) # multiplier from [0,1] to [0,255]
    wandb.log({"global_step": global_step, "valid/avg_rate": avg_rate}) 
    wandb.log({"global_step": global_step, "valid/loss": avg_loss})


## Training Loop
counter = 0
for epoch in range(0, 2_000):
    for input_img in tqdm(train_dataloader):
        optim.zero_grad()

        x = input_img.to(device)                     # [B,  3, 256, 256]
        y = compression_encoder(x)                   # [B, 12,  64,  64]
        y_hat = integer_quantization_class.apply(y)  # quantize to integer; custom straight-through gradient
        x_hat = compression_decoder(y_hat)           # [B,  3, 256, 256]

        distortion_loss = MSE(x, x_hat)
        rate_loss = rate_loss_fun(y)

        loss = 0.05 * distortion_loss + rate_loss
        loss.backward()
        optim.step()

        ## Run Validation, does not run at start
        if ((counter+1)%10_000) == 0:
            run_validation(valid_dataloader, counter)
        
        ## Logging Functionality
        if (counter%10) == 0:
            # log metrics to wandb
            wandb.log({"global_step": counter, "train/MSE": 255.0*255.0*distortion_loss}) # multiplier from [0,1] to [0,255]
            wandb.log({"global_step": counter, "train/rate_loss": rate_loss})
            wandb.log({"global_step": counter, "train/loss": loss})
            
            # images are too large, track every 50th step
            if (counter%50) == 0:
                x_and_xhat = torch.cat((x, x_hat), dim=3).permute(0,2,3,1)  # [B,C,H,W] -> [B,H,W,C]
                x_and_xhat = x_and_xhat[0,:]                                # [B,H,W,C] -> [H,W,C]
                x_and_xhat = x_and_xhat.cpu().detach().numpy()              # send to cpu & numpy
                image = wandb.Image(x_and_xhat, caption=f"original and compressed image")
                wandb.log({"global_step": counter, "train/images": image})
        
        counter += 1

