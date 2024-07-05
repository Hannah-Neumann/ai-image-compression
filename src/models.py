from torch import nn
from . import utils

## Description of Script
# This script defines a set of PyTorch modules to implement encoder-decoder architectures for image processing.
# It includes classes for two types of encoders (Encoder, HyperEncoder), decoders (Decoder, HyperDecoder), and their respective forward methods.
# Each encoder and decoder pair is designed to transform images into a latent space and reconstruct them,
# respectively. The network uses convolutional layers with varying strides to manage the downsampling and upsampling processes.

class Encoder(nn.Module):
    """Encoder that compresses an image into a lower-dimensional latent space."""
    def __init__(self):
        super().__init__()

        # Define the integer quantization operation
        self.integer_quantization = utils.IntegerQuantization.apply

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3,   out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=12,  kernel_size=5, stride=1, padding=2)
        
        # Activation functions
        self.act1 = nn.ReLU(inplace=False)
        self.act2 = nn.ReLU(inplace=False)
        self.act3 = nn.ReLU(inplace=False)
        self.act4 = nn.ReLU(inplace=False)
        self.act5 = nn.ReLU(inplace=False)
        self.act6 = nn.Tanh()

    def forward(self, x):
        # Forward pass through layers with activations
        layer1 = self.act1(self.conv1(x))       # [B,   3, 256, 256] --> [B, 128, 256, 256]
        layer2 = self.act2(self.conv2(layer1))  # [B, 128, 256, 256] --> [B, 256, 128, 128], downsample!
        layer3 = self.act3(self.conv3(layer2))  # [B, 256, 128, 128] --> [B, 256, 128, 128]
        layer4 = self.act4(self.conv4(layer3))  # [B, 256, 128, 128] --> [B, 512,  64,  64], downsample!
        layer5 = self.act5(self.conv5(layer4))  # [B, 512,  64,  64] --> [B, 512,  64,  64]
        layer6 = self.act6(self.conv6(layer5))  # [B, 512,  64,  64] --> [B,  12,  64,  64]
        y = 32.0 * layer6                       # Scale output for quantization: [-1,+1] --> [-32, +32]
        return y

class HyperEncoder(nn.Module):
    """Specialized encoder for processing output from the standard encoder."""
    def __init__(self):
        super().__init__()

        # Define the integer quantization operation
        self.integer_quantization = utils.IntegerQuantization.apply
        
        # Similar layer setup to Encoder but starts with 12 channels
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
        # Forward pass through layers with activations
        layer1 = self.act1(self.conv1(x))       # [B,   3, 64, 64] --> [B, 128, 64, 64]
        layer2 = self.act2(self.conv2(layer1))  # [B, 128, 64, 64] --> [B, 256, 32, 32], downsample!
        layer3 = self.act3(self.conv3(layer2))  # [B, 256, 32, 32] --> [B, 256, 32, 32]
        layer4 = self.act4(self.conv4(layer3))  # [B, 256, 32, 32] --> [B, 512, 16, 16], downsample!
        layer5 = self.act5(self.conv5(layer4))  # [B, 512, 16, 16] --> [B, 512, 16, 16]
        layer6 = self.act6(self.conv6(layer5))  # [B, 512, 16, 16] --> [B,  12, 16, 16]
        z = 16.0 * layer6                       # Scale output for quantization: [-1,+1] --> [-16, +16]
        return z

class Decoder(nn.Module):
    """Decoder that reconstructs the image from the latent space produced by Encoder."""
    def __init__(self):
        super().__init__()
        # Convolutional and Transposed Convolutional layers for reconstruction
        self.conv1 = nn.Conv2d(in_channels=12,  out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128,  kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=3,  kernel_size=3, stride=1, padding=1)
        
        # Activation functions
        self.act1 = nn.ReLU(inplace=False)
        self.act2 = nn.ReLU(inplace=False)
        self.act3 = nn.ReLU(inplace=False)
        self.act4 = nn.ReLU(inplace=False)
        self.act5 = nn.ReLU(inplace=False)
        self.act6 = nn.ReLU(inplace=False)
        self.act7 = nn.Sigmoid()  # Ensures output is between 0 and 1

    def forward(self, y):
        # Forward pass reconstructing the image from latent space
        layer1 = self.act1(self.conv1(y))       # [B,  12,  64,  64] --> [B, 512,  64,  64]
        layer2 = self.act2(self.conv2(layer1))  # [B, 512,  64,  64] --> [B, 512,  64,  64]
        layer3 = self.act3(self.conv3(layer2))  # [B, 512,  64,  64] --> [B, 256, 128, 128], upsample
        layer4 = self.act4(self.conv4(layer3))  # [B, 256, 128, 128] --> [B, 256, 128, 128]
        layer5 = self.act5(self.conv5(layer4))  # [B, 256, 128, 128] --> [B, 128, 256, 256], upsample
        layer6 = self.act6(self.conv6(layer5))  # [B, 128, 256, 256] --> [B, 128, 256, 256]
        layer7 = self.act7(self.conv7(layer6))  # [B, 128, 256, 256] --> [B,   3, 256, 256]
        x_hat = layer7                          # Final reconstructed image
        return x_hat

class HyperDecoder(nn.Module):
    """Specialized decoder for processing output from the hyperlatents to the probability parameters of the encoder entropy model."""
    def __init__(self):
        super().__init__()
        # Convolutional and Transposed Convolutional layers for reconstruction
        self.conv1 = nn.Conv2d(in_channels=12,  out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128,  kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=2*12, kernel_size=3, stride=1, padding=1)
        
        # Activation functions
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
        layer5 = self.act5(self.conv5(layer4))  # [B, 256, 32, 32] --> [B, 128, 64, 64], upsample
        layer6 = self.act6(self.conv6(layer5))  # [B, 128, 64, 64] --> [B, 128, 64, 64]
        layer7 = self.conv7(layer6)             # [B, 128, 64, 64] --> [B,  24, 64, 64]
        y_para = layer7
        return y_para