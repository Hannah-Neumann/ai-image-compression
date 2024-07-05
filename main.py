import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from src import utils
from src import dataset_utils
from src import losses
from src import models

def main():
    # Paths to the datasets
    dataset_train_path = "datasets/train_CLIC2021/"
    dataset_validation_path = "datasets/valid_CLIC2021/"

    # Determine the best device to run the training (e.g., 'cuda', 'mps', or 'cpu')
    device = utils.get_device()

    # Obtain transformations for training and validation datasets
    my_transforms_training = dataset_utils.get_training_transformation()
    my_transforms_validation = dataset_utils.get_validation_transformation()

    # Create dataset objects for training and validation
    training_data = dataset_utils.TrainImageDataset(img_dir=dataset_train_path, transform=my_transforms_training)
    validation_data = dataset_utils.ValidImageDataset(img_dir=dataset_validation_path, transform=my_transforms_validation)

    # Configure DataLoader keyword arguments for training and validation
    train_kwargs = utils.get_train_kwargs(device, batch_size=1, pin_memory=True, shuffle=True, num_workers=2)
    valid_kwargs = utils.get_valid_kwargs(device, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)

    # Set up DataLoaders for training and validation datasets
    train_dataloader = DataLoader(training_data, **train_kwargs)
    valid_dataloader = DataLoader(validation_data, **valid_kwargs)

    # Print out configuration to verify settings
    print(f"Training on device: {device}")
    print(f"Training DataLoader setup: {train_kwargs}")
    print(f"Validation DataLoader setup: {valid_kwargs}\n")
    

    # Initialize model components and move them to the appropriate device
    compression_encoder = models.Encoder().to(device)
    compression_hyper_encoder = models.HyperEncoder().to(device)
    compression_decoder = models.Decoder().to(device)
    compression_hyper_decoder = models.HyperDecoder().to(device)

    # Define the learning rate
    learning_rate = 1e-4

    # Initialize the optimizer with parameters from all models
    params = (
        list(compression_encoder.parameters()) + 
        list(compression_decoder.parameters()) +
        list(compression_hyper_encoder.parameters()) +
        list(compression_hyper_decoder.parameters())
    )
    optim = torch.optim.Adam(params, lr=learning_rate)

    # Print out configuration to verify settings
    print("Models have been initialized and moved to the specified device.")
    print("Integer quantization method is ready for use.")
    print(f"Learning rate is set to {learning_rate}.")
    print("Optimizer has been initialized with model parameters.")


    # WandB Tracking Initialization
    config_params = {
        "learning_rate": learning_rate,
        "architecture": "AI Image Compression AutoEncoder",
        "dataset": "CLIC_2021",
        "epochs": 2000
    }

    utils.initialize_wandb(project_name="AI-Image-Compression",
                           run_name="run_027",
                           config_params=config_params)
    

    # Set Training Parameters
    counter = 0
    epoch = 0
    max_epochs = 4_000

    # Decide whether to load a pre-trained model
    load_model = False
    load_path = "saved_models/compression_model_2000000.pt"

    # Dictionary to hold model components
    model_components = {'compression_encoder': compression_encoder,
                        'compression_hyper_encoder': compression_hyper_encoder,
                        'compression_decoder': compression_decoder,
                        'compression_hyper_decoder': compression_hyper_decoder,
                        'optimizer': optim}
    
    # Load model from checkpoint if applicable
    loaded_counter, loaded_epoch = utils.load_model_from_checkpoint(load_model, load_path, model_components)

    # Update training parameters if a model was successfully loaded
    if loaded_counter is not None:
        counter = loaded_counter
        epoch = loaded_epoch
        print(f"Resuming from epoch {loaded_epoch} and counter {loaded_counter}.")
    

    # Start of Trainings Loop
    while epoch < max_epochs:
        for input_img in tqdm(train_dataloader):

            # Adjust learning rate based on the training counter
            utils.adjust_learning_rate(optim, learning_rate, counter)

            optim.zero_grad(set_to_none=True)

            # Perform the forward pass
            x, x_hat, y, z, mu_y, sigma_y = utils.forward_pass(input_img,
                                                               compression_encoder,
                                                               compression_hyper_encoder,
                                                               compression_hyper_decoder,
                                                               compression_decoder,
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
            
            # Calculate gradients and perform backward pass
            loss.backward()
            optim.step()

            ## Run Validation; this does not run at start
            if ((counter+1)%20_000) == 0:
                print("Running validation loop")
                utils.run_validation(valid_dataloader,
                                     counter,
                                     compression_encoder,
                                     compression_hyper_encoder,
                                     compression_hyper_decoder,
                                     compression_decoder,
                                     device)

            # Save model checkpoint
            utils.save_model_checkpoint(counter, epoch, model_components, save_interval=200_000)

            # Set up Wandb logging
            metrics = {"train/MSE": 255.0 * 255.0 * distortion_loss.item(),  # Adjust for scale
                       "train/bpp_y": bpp_y,
                       "train/bpp_z": bpp_z,
                       "train/bpp_total": bpp_total,
                       "train/loss": loss.item()}

            # Log metrics and images to wandb
            utils.log_to_wandb(counter, metrics, images=(x, x_hat), image_step=5_000, log_step=500)

            counter += 1
        epoch += 1


if __name__ == '__main__':
    main()