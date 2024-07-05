import torch
import wandb
import torch.nn.functional as F
from src import utils

def mse_loss(x, x_hat):
    """
    Calculate the mean square error loss; our image quality loss.
    
    Args:
    x (torch.Tensor): The original image tensor.
    x_hat (torch.Tensor): The reconstructed image tensor after some transformation.

    Returns:
    torch.Tensor: The mean square error loss.
    """
    mse_val = torch.mean((x - x_hat) ** 2)
    return mse_val

def rate_loss(y, mu=0, sigma=5):
    """
    Calculate the rate loss using cross-entropy, which assumes a normal distribution.

    Args:
    y (torch.Tensor): The tensor for which the likelihood is calculated.
    mu (float, optional): The mean of the normal distribution. Defaults to 0.
    sigma (float, optional): The standard deviation of the normal distribution. Defaults to 5.

    Returns:
    torch.Tensor: The file size in bits as the negative sum of the log2 likelihoods.
    """
    likelihood = utils.get_likelihood(y, mu, sigma)
    filesize_in_bits = -torch.sum(torch.log2(likelihood))
    return filesize_in_bits