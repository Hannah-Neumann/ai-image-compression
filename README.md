
# AI Image Compression

## Introduction
This project implements the AI Image Compression pipeline as described in the papers "Joint autoregressive and hierarchical priors for learned image compression" (Ballé, 2018) and "Variational image compression with a scale hyperprior" (Ballé, 2018). The implementation aims to explore advanced image compression techniques using deep learning, as these papers are considered foundational work in the field.

As a product manager at Deep Render, I undertook this project to deepen my understanding of the technical and engineering aspects of image compression technologies. **Note**: This project is not affiliated with Deep Render and was conducted during my personal time.

## Results & Findings
The model was trained for 2 million iterations and the performance was tracked using Weights & Biases (Wandb). Here are the results obtained on the CLIC2021 validation dataset:

- **Mean Squared Error (MSE):** 
- **Bits Per Pixel (bpp):** 

![image_mse](images/Validation-2.png)
![image_bpp](images/Validation-3.png)

These results demonstrate improvements over traditional JPEG image compression techniques. Below are some visualizations from the training and validation phases as tracked on Wandb:

![image_picture_comparison_train](images/Train-4.png)
![image_picture_comparison_valid](images/Validation-4.png)

## How to Run the Code
To test the AI image compression model on your own data, follow these steps:

1. **Enable Model Loading:**
   Change the `load_model` flag from `False` to `True` in the script to use the pre-trained model.
   
2. **Specify Model Path:**
   Update the script to point to the trained model saved after 2M iterations.

3. **Input Image Specifications:**
   Ensure that the input image path is specified correctly in the script.

4. **Output:**
   The decompressed image (`xhat`) will be saved automatically. The compression efficiency can be calculated using the `bpp_total` which represents the total bits per pixel, calculated as `bpp_total * height * width` of the input image.

## Requirements
To run this project, ensure you have the following installed:
- `PyTorch`: For model creation and training.
- `Weights & Biases (wandb)`: For tracking experiments and results.
- `CUDA`: Optional, for GPU acceleration if available.
- `tqdm`: For displaying progress bars during training and validation.

## Installation
Before running the code, install the required Python packages using:
```bash
pip install torch wandb tqdm
```
## Sources
- Ballé, J. (2018). **"Joint Autoregressive and Hierarchical Priors for Learned Image Compression"**. Retrieved from [link to the paper](https://arxiv.org/pdf/1809.02736)
- Ballé, J. (2018). **"Variational Image Compression with a Scale Hyperprior"**. Retrieved from [link to the paper](https://arxiv.org/pdf/1802.01436v2)

Additional resources and tools utilized in this project include:

- **Weights & Biases (Wandb)** - Tool for tracking experiments, used for monitoring model training and validation. More information available at [wandb.com](https://wandb.com).
- **PyTorch** - The deep learning framework used for model implementation. Documentation and more details can be found at [pytorch.org](https://pytorch.org).
- **CLIC2021 Dataset** - The dataset used for training and validation, details available at [CLIC website](https://www.compression.cc/2021/).