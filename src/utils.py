import torch
import wandb
from . import losses
from tqdm import tqdm

def get_device(override=None):
    """
    Determine the most suitable device for PyTorch operations based on availability.

    This function checks for the availability of CUDA and MPS (Metal Performance Shaders on macOS)
    and selects the most appropriate computing device. It allows for an optional override to 
    specify a device manually.

    Parameters:
        override (str, optional): Manually specify 'cuda', 'mps', or 'cpu' to override the automatic detection.

    Returns:
        torch.device: The selected device.
    """
    # Check availability of CUDA and MPS
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    
    # Determine the best available device
    if override:
        device = torch.device(override)
    elif use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    return device

def get_train_kwargs(device, batch_size=1, pin_memory=True, shuffle=True, num_workers=2):
    """
    Generate keyword arguments for DataLoader configuration during training.

    This function constructs a dictionary of DataLoader arguments optimized based on the
    device (CPU or CUDA) used for training. Additional parameters are provided for 
    customization based on user preferences.

    Parameters:
        device (str): Device type, e.g., 'cuda' or 'cpu'.
        batch_size (int, optional): Number of samples in each batch. Defaults to 1.
        pin_memory (bool, optional): Whether to use pinned memory. Recommended when using GPUs. Defaults to True.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 2.

    Returns:
        dict: A dictionary containing DataLoader keyword arguments.
    """
    # Common kwargs for any device
    train_kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'pin_memory': pin_memory}

    # Additional kwargs for CUDA to utilize more workers for efficient data loading
    if device == 'cuda':
        train_kwargs['num_workers'] = num_workers

    return train_kwargs

def get_valid_kwargs(device, batch_size=1, pin_memory=True, shuffle=False, num_workers=2):
    """
    Generate keyword arguments for DataLoader configuration during validation.

    Constructs a dictionary of DataLoader arguments optimized based on the device
    used for validation (CPU or CUDA), with options for customization according to
    user needs. 

    Parameters:
        device (str): Device type, e.g., 'cuda' or 'cpu'.
        batch_size (int, optional): Number of samples in each batch. Defaults to 1.
        pin_memory (bool, optional): Whether to use pinned memory. Recommended when using GPUs. Defaults to True.
        shuffle (bool, optional): Whether to shuffle the data. Generally false for validation. Defaults to False.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 2.

    Returns:
        dict: A dictionary containing DataLoader keyword arguments.
    """
    # Common kwargs for any device
    valid_kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'pin_memory': pin_memory}

    # Additional kwargs for CUDA to utilize more workers for efficient data loading
    if device == 'cuda':
        valid_kwargs['num_workers'] = num_workers

    return valid_kwargs
    
def get_likelihood(y, mu, sigma, eps=1e-7):
    """
    Calculate the likelihood of the data assuming a normal distribution (PMF), modified by added noise.
    This is to model the discrete data through a relaxation of the normal probability mass function.

    Args:
    y (torch.Tensor): The data tensor.
    mu (float): Mean of the normal distribution.
    sigma (float): Standard deviation of the normal distribution.
    eps (float, optional): A small epsilon value to prevent log(0). Defaults to 1e-7.

    Returns:
    torch.Tensor: The likelihood values for each element in y.
    """
    # Add uniform noise centered around 0 with range [-0.5, 0.5]
    y_noisy = y + (torch.randn_like(y, device=y.device) - 0.5)

    # Define the normal distribution and compute the likelihood using the CDF
    normal_dist = torch.distributions.Normal(mu, sigma)
    likelihood = normal_dist.cdf(y_noisy + 0.5) - normal_dist.cdf(y_noisy - 0.5)
    
    # Ensure no probability is zero
    likelihood = torch.clamp(likelihood, min=eps)

    return likelihood

class IntegerQuantization(torch.autograd.Function):
    """
    A custom autograd Function for integer quantization, using straight-through estimator for backpropagation.
    """
    @staticmethod
    def forward(ctx, y):
        """
        In the forward pass we receive a Tensor containing the input and return a Tensor containing the output.
        ctx is a context object that can be used to stash information for backward computation.
        """
        ctx.save_for_backward(y)
        y_quantized = torch.round(y)  # Perform quantization by rounding to the nearest integer
        return y_quantized

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive the gradient of the loss with respect to the output,
        and we need to compute the gradient of the loss with respect to the input.
        """
        y, = ctx.saved_tensors
        grad_input = grad_output.clone()  # Straight-through estimator: grad_input is passed unchanged
        return grad_input

def initialize_wandb(project_name, run_name, config_params):
    """
    Initializes a wandb run with the specified project, run name, and configuration.

    Parameters:
    - project_name (str): Name of the wandb project.
    - run_name (str): Name of the specific run within the project.
    - config_params (dict): Dictionary containing configuration parameters such as learning rate,
      architecture description, dataset used, number of epochs, etc.

    This function starts a new wandb run and defines a global step metric. It prints out the initialization
    status and the configuration used for easy debugging and tracking.
    
    If initialization fails, it catches the exception and prints an error message.
    """
    try:
        wandb.init(project=project_name, name=run_name, config=config_params)
        wandb.define_metric("*", step_metric="global_step")
        print(f"W&B initialized for project '{project_name}' with run name '{run_name}'.")
        print("Configuration:")
        for key, value in config_params.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Failed to initialize W&B: {str(e)}")

def load_model_from_checkpoint(load_model, load_path, model_components):
    """
    Conditionally loads model components from a checkpoint file if load_model is True.

    Parameters:
    - load_model (bool): Flag to determine whether to load the model.
    - load_path (str): File path from which the model checkpoint is loaded.
    - model_components (dict): Dictionary mapping component names to their corresponding PyTorch models.
      This dictionary should include the model parts that require loading, such as encoders, decoders, and optimizers.

    Returns:
    - Tuple of (counter, epoch) if the model is loaded successfully; both are None if not loaded or on failure.
      'counter' is an integer indicating the last training step completed,
      'epoch' is an integer indicating the last epoch completed.

    Prints the path from which the model was loaded on success, or an error message on failure.
    """
    if load_model:
        try:
            checkpoint = torch.load(load_path)
            model_components['compression_encoder'].load_state_dict(checkpoint['compression_encoder_state_dict'])
            model_components['compression_hyper_encoder'].load_state_dict(checkpoint['compression_hyper_encoder_state_dict'])
            model_components['compression_decoder'].load_state_dict(checkpoint['compression_decoder_state_dict'])
            model_components['compression_hyper_decoder'].load_state_dict(checkpoint['compression_hyper_decoder_state_dict'])
            model_components['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
            counter = checkpoint['counter']
            epoch = checkpoint['epoch']
            print(f"\n\nModel loaded from path: {load_path}\n\n")
            return counter, epoch
        except Exception as e:
            print(f"Failed to load the model from path {load_path}: {str(e)}")
            return None, None
    else:
        return None, None

def adjust_learning_rate(optimizer, learning_rate, counter):
    """
    Adjusts the learning rate based on the training counter.

    Parameters:
    - optimizer (torch.optim.Optimizer): The optimizer for which the learning rate will be adjusted.
    - learning_rate (float): The base learning rate.
    - counter (int): The current step counter in the training loop.

    Adjusts the learning rate at specific training milestones.
    """
    if counter == 1_500_000:
        new_lr = 0.10 * learning_rate
        for g in optimizer.param_groups:
            g['lr'] = new_lr
        print(f"Learning rate reduced to {new_lr}")

    if counter == 1_800_000:
        new_lr = 0.01 * learning_rate
        for g in optimizer.param_groups:
            g['lr'] = new_lr
        print(f"Learning rate further reduced to {new_lr}")

def forward_pass(input_img, encoder, hyper_encoder, hyper_decoder, decoder, device):
    """Perform the forward pass of the compression model."""
    x = input_img.to(device)   # Move input to the correct device
    y = encoder(x)             # Encode input image
    z = hyper_encoder(y)       # Hyper encode the features

    z_hat = hyper_encoder.integer_quantization(z)   # Quantize hyper features
    y_para = hyper_decoder(z_hat)                   # Decode hyper features

    mu_y = y_para[:, 0:12]                          # Mean prediction
    sigma_y = 0.01 + torch.abs(y_para[:, 12:24])    # Ensure non-negative sigma

    # Quantize only the residuals after the mu prediction
    y_res = y - mu_y
    y_hat_res = encoder.integer_quantization(y_res) # Quantize features
    y_hat = y_hat_res + mu_y

    x_hat = decoder(y_hat)     # Decode features to reconstruct image
    return x, x_hat, y, z, mu_y, sigma_y

def save_model_checkpoint(counter, epoch, model_components, save_interval=200000, base_path='saved_models'):
    """
    Saves the model checkpoint at specified intervals.

    Parameters:
    - counter (int): The current training counter.
    - epoch (int): The current epoch.
    - model_components (dict): A dictionary containing the model components to save.
    - save_interval (int): The interval at which to save the model.
    - base_path (str): The base directory to save the model files.
    
    The function checks if the counter is at a save point and if so, saves the model.
    """
    if counter % save_interval == 0:
        save_path = f"{base_path}/compression_model_{counter}.pt"
        torch.save({
            'compression_encoder_state_dict': model_components['compression_encoder'].state_dict(),
            'compression_hyper_encoder_state_dict': model_components['compression_hyper_encoder'].state_dict(),
            'compression_decoder_state_dict': model_components['compression_decoder'].state_dict(),
            'compression_hyper_decoder_state_dict': model_components['compression_hyper_decoder'].state_dict(),
            'optimizer_state_dict': model_components['optimizer'].state_dict(),
            'counter': counter,
            'epoch': epoch
        }, save_path)
        print(f"Model checkpoint saved to {save_path}")

def log_to_wandb(counter, metrics, images=None, image_step=5000, log_step=500):
    """
    Logs metrics and images to wandb at specified intervals.

    Parameters:
    - counter (int): The current step counter in the training loop.
    - metrics (dict): A dictionary of metrics to log.
    - images (tuple of torch.Tensor): Tuple containing original and reconstructed images (optional).
    - image_step (int): Interval for logging images.
    - log_step (int): Interval for logging metrics.

    Metrics are logged every 'log_step' steps, and images are logged every 'image_step' steps.
    """
    if counter % log_step == 0:
        wandb.log({"global_step": counter, **metrics})

        if images is not None and counter % image_step == 0:
            original_img, reconstructed_img = images
            # Concatenate images horizontally and adjust image data
            combined_image = torch.cat((original_img, reconstructed_img), dim=3).permute(0, 2, 3, 1)
            combined_image = combined_image[0]  # Take first image in batch for logging
            combined_image = torch.clamp(combined_image, 0.0, 1.0)  # Clamp values to be between 0 and 1
            combined_image = combined_image.cpu().detach().numpy()  # Convert to numpy array for logging

            image = wandb.Image(combined_image, caption="Original and Compressed Image")
            wandb.log({"global_step": counter, "train/images": image})

def log_validation_images(x, x_hat, val_loop_counter, global_step, distortion_loss):
    """ Helper function to log validation images to wandb. """
    combined_image = torch.cat((x, x_hat), dim=3).permute(0,2,3,1)  # Concatenate and permute dimensions
    combined_image = torch.clamp(combined_image, 0, 1)              # Clamp to valid image range [0, 1]
    combined_image = combined_image[0].cpu().detach().numpy()       # Convert to numpy
    caption = f"Original and Compressed Image, MSE: {255.0 * 255.0 * distortion_loss.item()}"
    image = wandb.Image(combined_image, caption=caption)
    wandb.log({f"global_step": global_step, f"valid/images_{val_loop_counter}": image})

def run_validation(valid_dataloader, global_step, encoder, hyper_encoder, hyper_decoder, decoder, device):
    """ Run validation on the validation dataset and log metrics and images using wandb. """
    val_counter = 0.0
    metrics = {'distortion': 0.0, 'bpp_total': 0.0, 'bpp_y': 0.0, 'bpp_z': 0.0, 'loss': 0.0}

    for val_loop_counter, input_img in enumerate(tqdm(valid_dataloader)):
        with torch.no_grad():
            # Perform the forward pass
            x, x_hat, y, z, mu_y, sigma_y = forward_pass(input_img,
                                                         encoder,
                                                         hyper_encoder,
                                                         hyper_decoder,
                                                         decoder,
                                                         device)
            # Calculate the losses
            spatial_px = x.shape[2] * x.shape[3]
            distortion_loss = losses.mse_loss(x, x_hat)
            rate_loss_y = losses.rate_loss(y, mu_y, sigma_y)
            rate_loss_z = losses.rate_loss(z)
            bpp_y = rate_loss_y / spatial_px
            bpp_z = rate_loss_z / spatial_px
            bpp_total = bpp_y + bpp_z
            loss = 1000.0 * distortion_loss + bpp_total

            for key, value in [('distortion', distortion_loss), ('bpp_total', bpp_total), ('bpp_y', bpp_y), ('bpp_z', bpp_z), ('loss', loss)]:
                metrics[key] += value.item()

            log_validation_images(x, x_hat, val_loop_counter, global_step, distortion_loss)

            val_counter += 1

    # Log aggregated validation metrics
    for metric in metrics:
        metrics[metric] /= val_counter
        wandb.log({f"global_step": global_step, f"valid/{metric}": metrics[metric] * (255.0**2 if metric == 'distortion' else 1)})