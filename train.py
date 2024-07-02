import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import v2
import matplotlib.pyplot as plt

import wandb
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

## Helper Functions
def MSE(x, x_hat):
    mse_val = torch.mean((x - x_hat)**2)
    return mse_val 

# Rate Loss, Assumes a Normal Distribution (PMF)
def rate_loss_fun(y, mu=0, sigma=5):
    likelihood = get_likelihood(y, mu, sigma)
    filesize_in_bits = -torch.sum(torch.log2(likelihood))

    return filesize_in_bits

def get_likelihood(y, mu, sigma, eps=1e-7):
    y_noisy = y + (torch.randn_like(y, device='cuda') - 0.50)  # add uniform noise between [-0.5, 0.50]
    
    normal_dist = torch.distributions.normal.Normal(mu, sigma)
    likelihood = normal_dist.cdf(y_noisy + 0.50) - normal_dist.cdf(y_noisy - 0.50)
    
    # probability is not allowed to be zero
    likelihood = F.threshold(likelihood, eps, eps)

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
        self.image_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]*50
        self.length = len(self.image_files)

    def __len__(self):
        return self.length

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

class HyperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=12,  out_channels=128, kernel_size=3, stride=1, padding=1)
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
        layer1 = self.act1(self.conv1(x))       # [B,   3, 64, 64] --> [B, 128, 64, 64]

        layer2 = self.act2(self.conv2(layer1))  # [B, 128, 64, 64] --> [B, 256, 32, 32], downsample!
        layer3 = self.act3(self.conv3(layer2))  # [B, 256, 32, 32] --> [B, 256, 32, 32]

        layer4 = self.act4(self.conv4(layer3))  # [B, 256, 32, 32] --> [B, 512, 16, 16], downsample!
        layer5 = self.act5(self.conv5(layer4))  # [B, 512, 16, 16] --> [B, 512, 16, 16]

        layer6 = self.act6(self.conv6(layer5))  # [B, 512, 16, 16] --> [B,  12, 16, 16]
        z      = 16.0 * layer6                  # brings data from [-1,+1] to [-16, +16]

        return z

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

class HyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=12,  out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)

        self.conv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128,  kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(in_channels=128, out_channels=2*12, kernel_size=3, stride=1, padding=1)
        
        self.act1 = nn.ReLU(inplace=False)
        self.act2 = nn.ReLU(inplace=False)
        self.act3 = nn.ReLU(inplace=False)
        self.act4 = nn.ReLU(inplace=False)
        self.act5 = nn.ReLU(inplace=False)
        self.act6 = nn.ReLU(inplace=False)

    def forward(self, z):
        layer1 = self.act1(self.conv1(z))       # [B,  12, 16, 16] --> [B, 512, 16, 16]
        layer2 = self.act2(self.conv2(layer1))  # [B, 512, 16, 16] --> [B, 512, 16, 16]

        layer3 = self.act3(self.conv3(layer2))  # [B, 512, 16, 16] --> [B, 256, 32, 32], upsample
        layer4 = self.act4(self.conv4(layer3))  # [B, 256, 32, 32] --> [B, 256, 32, 32]

        layer5 = self.act5(self.conv5(layer4))  # [B, 256, 32, 32] --> [B, 128, 64, 64]
        layer6 = self.act6(self.conv6(layer5))  # [B, 128, 64, 64] --> [B, 128, 64, 64]
        layer7 = self.conv7(layer6)             # [B, 128, 64, 64] --> [B,  24, 64, 64]
        y_para = layer7

        return y_para

def main():
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
                                        img_divisible_by = 16,
                                        transform=my_transforms_validation)
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
        cuda_train_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': True}
        cuda_valid_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False}
        train_kwargs.update(cuda_train_kwargs)
        valid_kwargs.update(cuda_valid_kwargs)

    print("\n\nDevice: ", device, "\n\n")
    train_dataloader = DataLoader(training_data, **train_kwargs)
    valid_dataloader = DataLoader(validation_data, **valid_kwargs)
                                
    learning_rate = 1e-4

    compression_encoder = Encoder().to(device)
    compression_hyper_encoder = HyperEncoder().to(device)
    compression_decoder = Decoder().to(device)
    compression_hyper_decoder = HyperDecoder().to(device)
    integer_quantization = integer_quantization_class.apply
    
    optim = torch.optim.Adam([*compression_encoder.parameters(), *compression_decoder.parameters(),
                            *compression_hyper_encoder.parameters(), *compression_hyper_decoder.parameters()],
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
            "epochs": 2_000,
            })
    wandb.define_metric("*", step_metric="global_step")


    ## Validation Lopp:
    def run_validation(valid_dataloader, global_step):
        val_counter = 0.0
        avg_distortion = 0.0
        avg_bpp_total = 0.0
        avg_bpp_y = 0.0
        avg_bpp_z = 0.0
        avg_loss = 0.0

        for val_loop_counter, input_img in enumerate(tqdm(valid_dataloader)):
            with torch.no_grad():
                x = input_img.to(device)                      # [B,  3,     H,     W]
                y = compression_encoder(x)                    # [B, 12,  H//4,  W//4]
                z = compression_hyper_encoder(y)              # [B, 12, H//16, W//16]

                z_hat  = integer_quantization(z)  # quantize to integer; custom straight-through gradient
                y_para = compression_hyper_decoder(z_hat)     # [B, 24,  H//4,  W//4]
                y_mu = y_para[:,0:12]                         # [B, 12,  H//4,  W//4]
                y_sigma = 0.01 + torch.abs(y_para[:,12:24])   # [B, 12,  H//4,  W//4]; NO NEGATIVE SIGMA

                # Create y_hat, quantize only the residual after the mu prediction
                y_res = y - y_mu
                y_res_hat = integer_quantization(y_res) # quantize to integer; custom straight-through gradient
                y_hat = y_res_hat + y_mu

                x_hat = compression_decoder(y_hat)            # [B, 3, H, W]
                
                distortion_loss = MSE(x, x_hat)

                rate_loss_y = rate_loss_fun(y, y_mu, y_sigma)
                bpp_y = rate_loss_y / (1.0*x.shape[2]*x.shape[3])

                rate_loss_z = rate_loss_fun(z)
                bpp_z = rate_loss_z / (1.0*x.shape[2]*x.shape[3])

                bpp_total = bpp_y + bpp_z

                loss = 1000.0*distortion_loss + bpp_total

                avg_distortion += distortion_loss
                avg_bpp_total += bpp_total
                avg_bpp_y += bpp_y
                avg_bpp_z += bpp_z
                avg_loss += loss
                val_counter +=1
                
                # Log all Validation images:
                x_and_xhat = torch.cat((x, x_hat), dim=3).permute(0,2,3,1)  # [B,C,H,W] -> [B,H,W,C]
                x_and_xhat = x_and_xhat[0,:]                                # [B,H,W,C] -> [H,W,C]
                x_and_xhat = F.threshold(x_and_xhat, 0.0, 0.0)              # lower bound to 0.0
                x_and_xhat = -F.threshold(-x_and_xhat, -1.0, -1.0)          # uper bpund to 1.0
                x_and_xhat = x_and_xhat.cpu().detach().numpy()              # send to cpu & numpy
                image = wandb.Image(x_and_xhat, caption=f"original and compressed image, MSE: {255.0*255.0*distortion_loss}")
                wandb.log({"global_step": global_step, f"valid/images_{val_loop_counter}": image})
        
        # Log aggrregated validation metrics:
        avg_distortion = avg_distortion / val_counter
        avg_bpp_total = avg_bpp_total / val_counter
        avg_bpp_y = avg_bpp_y / val_counter
        avg_bpp_z = avg_bpp_z / val_counter
        avg_loss = avg_loss / val_counter
        wandb.log({"global_step": global_step, "valid/MSE": 255.0*255.0*avg_distortion}) # multiplier from [0,1] to [0,255]
        wandb.log({"global_step": global_step, "valid/avg_bpp_total": avg_bpp_total}) 
        wandb.log({"global_step": global_step, "valid/avg_bpp_y": avg_bpp_y}) 
        wandb.log({"global_step": global_step, "valid/avg_bpp_z": avg_bpp_z}) 
        wandb.log({"global_step": global_step, "valid/loss": avg_loss})


    ## Training Loop
    counter = 0
    epoch = 0
    max_epochs = 4_000

    ## OPTIONAL, LOAD MODEL
    load_model = False
    load_path = "saved_models/compression_model_2000000.pt"    

    if(load_model):
        checkpoint = torch.load(load_path)
        compression_encoder.load_state_dict(checkpoint['compression_encoder_state_dict'])
        compression_hyper_encoder.load_state_dict(checkpoint['compression_hyper_encoder_state_dict'])
        compression_decoder.load_state_dict(checkpoint['compression_decoder_state_dict'])
        compression_hyper_decoder.load_state_dict(checkpoint['compression_hyper_decoder_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        counter = checkpoint['counter']
        epoch = checkpoint['epoch']
        print("\n\nModel loaded, Path: ", load_path, "\n\n")

    while epoch < max_epochs:
        for input_img in tqdm(train_dataloader):

            # Reduce the learning rate in the later stages of training:
            if counter == 1_500_000:
                for g in optim.param_groups:
                    g['lr'] = 0.10*learning_rate
            if counter == 1_800_000:
                for g in optim.param_groups:
                    g['lr'] = 0.01*learning_rate

            optim.zero_grad(set_to_none=True)

            x = input_img.to(device)                      # [B,  3, 256, 256]
            y = compression_encoder(x)                    # [B, 12,  64,  64]
            z = compression_hyper_encoder(y)              # [B, 12,  16,  16]

            z_hat  = integer_quantization(z)  # quantize to integer; custom straight-through gradient
            y_para = compression_hyper_decoder(z_hat)      # [B, 24,  64,  64]
            y_mu = y_para[:,0:12]                         # [B, 12,  64,  64]
            y_sigma = 0.01 + torch.abs(y_para[:,12:24])   # [B, 12,  64,  64]; NO NEGATIVE SIGMA

            # Create y_hat, quantize only the residual after the mu prediction
            y_res = y - y_mu
            y_res_hat = integer_quantization(y_res) # quantize to integer; custom straight-through gradient
            y_hat = y_res_hat + y_mu

            x_hat = compression_decoder(y_hat)           # [B,  3, 256, 256]

            # Losses
            distortion_loss = MSE(x, x_hat)

            rate_loss_y = rate_loss_fun(y, y_mu, y_sigma)
            bpp_y = rate_loss_y / (1.0*x.shape[2]*x.shape[3])

            rate_loss_z = rate_loss_fun(z)
            bpp_z = rate_loss_z / (1.0*x.shape[2]*x.shape[3])

            bpp_total = bpp_y + bpp_z

            loss = 1000.0*distortion_loss + bpp_total
            loss.backward()
            optim.step()

            ## Run Validation, does not run at start
            if ((counter+1)%20_000) == 0:
                run_validation(valid_dataloader, counter)

            ## Save Model
            if ((counter+0)%200_000) == 0:
                torch.save({'compression_encoder_state_dict': compression_encoder.state_dict(),
                            'compression_hyper_encoder_state_dict': compression_hyper_encoder.state_dict(),
                            'compression_decoder_state_dict': compression_decoder.state_dict(),
                            'compression_hyper_decoder_state_dict': compression_hyper_decoder.state_dict(),
                            'optimizer_state_dict': optim.state_dict(),
                            'counter': counter,
                            'epoch': epoch,
                        }, f"saved_models/compression_model_{counter}.pt")

            ## Logging Functionality
            if (counter%500) == 0:
                # log metrics to wandb
                wandb.log({"global_step": counter, "train/MSE": 255.0*255.0*distortion_loss}) # multiplier from [0,1] to [0,255]
                wandb.log({"global_step": counter, "train/bpp_y": bpp_y})
                wandb.log({"global_step": counter, "train/bpp_z": bpp_z})
                wandb.log({"global_step": counter, "train/bpp_total": bpp_total})
                wandb.log({"global_step": counter, "train/loss": loss})
                
                # images are too large, track every 50th step
                if (counter%5_000) == 0:
                    x_and_xhat = torch.cat((x, x_hat), dim=3).permute(0,2,3,1)  # [B,C,H,W] -> [B,H,W,C]
                    x_and_xhat = x_and_xhat[0,:]                                # [B,H,W,C] -> [H,W,C]
                    x_and_xhat = F.threshold(x_and_xhat, 0.0, 0.0)              # lower bound to 0.0
                    x_and_xhat = -F.threshold(-x_and_xhat, -1.0, -1.0)          # uper bpund to 1.0
                    x_and_xhat = x_and_xhat.cpu().detach().numpy()              # send to cpu & numpy
                    image = wandb.Image(x_and_xhat, caption=f"original and compressed image")
                    wandb.log({"global_step": counter, "train/images": image})
            
            counter += 1
        epoch += 1

if __name__ == '__main__':
    main()
