import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

## Define Transformations for Training and Validation

def get_training_transformation(random_flip_probability=0.20, crop_size=(256, 256), dtype=torch.float32):
    """
    Define and return a composite transformation for training images.
    
    This function assembles multiple transformations to prepare images for the training process, making it suitable
    for a variety of machine learning models that require normalized and augmented image data.

    Parameters:
        random_flip_probability (float): Probability for flipping the image horizontally. Default is 0.20.
        crop_size (tuple): The target size (height, width) after cropping. Default is (256, 256).
        dtype (torch.dtype): Desired data type of the transformed images (e.g., torch.float32).

    Returns:
        torchvision.transforms.Compose: A composition of image transformations.
    
    The transformations included are:
    - `ToTensor`: Converts PIL images or numpy arrays to Torch Tensors, scaling pixel intensity values to [0, 1].
    - `RandomResizedCrop`: Randomly crops and resizes the image to the specified size with antialiasing to enhance image quality.
    - `RandomHorizontalFlip`: Randomly flips the image horizontally to augment the training dataset with the given probability.
    - `ConvertImageDtype`: Converts the Tensor data type to the specified `dtype`, facilitating compatibility with neural network inputs.
    """

    my_transforms_training = v2.Compose([v2.ToTensor(),
                                        v2.RandomResizedCrop(size=(256, 256), antialias=True),
                                        v2.RandomHorizontalFlip(p=0.20),
                                        v2.ToDtype(torch.float32, scale=True)
                                        ])
    return v2.Compose([
        v2.ToTensor(),
        v2.RandomResizedCrop(size=crop_size, antialias=True),
        v2.RandomHorizontalFlip(p=random_flip_probability),
        v2.ToDtype(torch.float32, scale=True)
    ])

def get_validation_transformation():
    """
    Define and return a basic transformation for validation images.
    
    This function sets up transformations for the validation dataset, which typically requires less augmentation:
    - `ToTensor`: Converts images to Torch Tensors with values scaled between 0 and 1.
    - `ConvertImageDtype`: Ensures the tensor is of type float32 for consistency with the model input requirements.
    
    Returns:
        torchvision.transforms.Compose: A composition of image transformations.
    """
    return v2.Compose([
        v2.ToTensor(),
        v2.ToDtype(torch.float32, scale=True)
    ])

## Custom Dataset Class for Training Images
class TrainImageDataset(Dataset):
    """Dataset class for training images."""
    def __init__(self, img_dir, transform=None):
        """
        Args:
        img_dir (str): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))] * 50
        self.length = len(self.image_files)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Fetch an image and apply transformations."""
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = read_image(img_path)

        # Convert grayscale images to RGB by duplicating the grayscale channel
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)  # More efficient than torch.cat

        if self.transform:
            image = self.transform(image)
        return image

## Custom Dataset Class for Validation Images
class ValidImageDataset(Dataset):
    """Dataset class for validation images."""
    def __init__(self, img_dir, img_divisible_by=16, transform=None):
        """
        Args:
        img_dir (str): Directory with all the images.
        img_divisible_by (int): Ensures the dimensions of the images are divisible by this value.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_divisible_by = img_divisible_by
        self.image_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Fetch an image, adjust its size, and apply transformations."""
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = read_image(img_path)

        # Resize image dimensions to be divisible by 'img_divisible_by'
        new_height = (image.shape[1] // self.img_divisible_by) * self.img_divisible_by
        new_width = (image.shape[2] // self.img_divisible_by) * self.img_divisible_by
        image = image[:, :new_height, :new_width]

        # Convert grayscale images to RGB
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if self.transform:
            image = self.transform(image)
        return image
