# Pix2Pix GAN for Satellite Image Segmentation

## Introduction
This project leverages the Pix2Pix GAN framework to perform image-to-image translation on satellite imagery. The goal is to convert satellite photos into segmented maps that highlight key features such as roads, buildings, and natural elements. The Pix2Pix model is particularly effective for this task due to its ability to generate high-quality, realistic images while preserving the structural integrity of the input data. This implementation is built using PyTorch and demonstrates a robust application of GANs for real-world geospatial analysis.

## Model Architecture

### Generator (U-Net)

**Architecture**:  
The generator is based on the U-Net architecture, a popular choice for image segmentation tasks. It features an encoder-decoder structure with skip connections:

- **Encoder**: Progressively downsamples the input image through a series of contracting blocks (convolutions followed by max-pooling), extracting high-level features and reducing spatial dimensions.
- **Decoder**: Upsamples the features back to the original resolution through expanding blocks, reconstructing the segmented map.
- **Skip Connections**: Concatenate feature maps from the encoder to the decoder at corresponding levels, preserving fine-grained spatial details critical for accurate segmentation.

**Implementation Details**:
- Uses 6 contracting blocks and 6 expanding blocks, with channel sizes doubling/halving at each level (starting from 32 hidden channels).
- Includes dropout in the initial contracting blocks for regularization and batch normalization for training stability.
- Final layer uses a 1x1 convolution to map features to the desired output channels (3 for RGB segmented maps), followed by a sigmoid activation.

**Purpose**:  
Generates realistic segmented maps from satellite images, balancing global context and local precision.

### Discriminator (PatchGAN)

**Architecture**:  
The discriminator employs a PatchGAN design, which evaluates overlapping patches of the image rather than the entire image:

- Takes as input a concatenation of the satellite image (condition) and either the real or generated segmented map.
- Consists of 4 contracting blocks, similar to the U-Net encoder, followed by a final 1x1 convolution to produce a grid of patch-wise real/fake predictions.
- Patch size is determined by the receptive field of the network (approximately 70x70 pixels in this implementation).

**Implementation Details**:
- Initial hidden channels set to 8, doubling with each contracting block.
- Uses LeakyReLU (slope 0.2) for non-linearity and avoids batch normalization in the first layer for stability.

**Purpose**:  
Classifies whether patches of the generated map are realistic, providing local feedback to the generator to improve texture and detail.

## Loss Functions

The Pix2Pix model is trained using a combination of two loss functions:

### Adversarial Loss
- Encourages the generator to produce images that fool the discriminator.
- Computed using `BCEWithLogitsLoss`, comparing discriminator outputs to target labels (1 for real, 0 for fake).
- Discriminator loss is the average of real and fake classification losses.

### Reconstruction Loss
- Ensures the generated map closely matches the ground truth map.
- Uses L1 loss (`L1Loss`) to measure pixel-wise differences between the generated and real images.
- Weighted by a hyperparameter `lambda_recon` to balance its contribution relative to the adversarial loss.

**Total Generator Loss**:  
Sum of the adversarial loss and the weighted reconstruction loss, driving the model to produce both realistic and accurate outputs.

## Hyperparameters

| Hyperparameter           | Value     | Description                                 |
|--------------------------|-----------|---------------------------------------------|
| Number of Epochs         | 20        | Total passes through the dataset            |
| Batch Size               | 4         | Number of images per training iteration     |
| Learning Rate            | 0.0002    | Step size for Adam optimizer                |
| Input Channels           | 3         | RGB channels for satellite images           |
| Output Channels          | 3         | RGB channels for segmented maps             |
| Lambda Reconstruction    | 200       | Weight of L1 reconstruction loss            |
| Target Image Size        | 256x256   | Resolution of input/output images           |
| Display Step             | 200       | Frequency of loss reporting and visualization |

## Dataset

The dataset used is the **"maps"** dataset, consisting of paired satellite images and their corresponding segmented maps.

**Structure**:
- Each image is a composite where the left half is the satellite image (condition) and the right half is the segmented map (real image).

**Preprocessing**:
- Images are resized to 256x256 pixels using interpolation (`nn.functional.interpolate`) to ensure uniform input size.
- Converted to tensors using `transforms.ToTensor()` for PyTorch compatibility.

**Loading**:
- Loaded via `torchvision.datasets.ImageFolder`, with a custom path on Google Drive.
- Batched using `DataLoader` with shuffling for training.

## Training Process

The model is trained using an alternating optimization scheme typical of GANs:

### Discriminator Update
- Trained to distinguish real segmented maps from generated ones.
- **Input**: Concatenation of the satellite image with either the real map or the generated map (detached).
- **Loss**: Average of `BCEWithLogitsLoss` for real (target 1) and fake (target 0) predictions.
- **Optimizer**: Adam with learning rate 0.0002.

### Generator Update
- Trained to fool the discriminator and minimize reconstruction error.
- **Loss**: Combination of adversarial loss (`BCEWithLogitsLoss` with target 1) and L1 reconstruction loss (weighted by 200).
- **Optimizer**: Adam with learning rate 0.0002.

**Initialization**:
- Weights are initialized with a normal distribution (mean 0, std 0.02) for convolutional and batch normalization layers.

**Progress Monitoring**:
- Losses and sample outputs (condition, real, and fake images) are visualized every 200 steps using a custom `show_tensor_images` function.

**Training Duration**:
- The training process runs for 20 epochs, refining the generator’s output quality over time.

## Results

> From left to right: input satellite image, generated segmented map, and ground truth map.

The generated maps demonstrate the model's capability to accurately segment key features (e.g., roads, buildings) while maintaining visual coherence with the input satellite images. The PatchGAN discriminator contributes to sharp, detailed outputs, and the U-Net generator ensures structural fidelity to the ground truth.

## How to Run the Code

### Install Dependencies
```bash
pip install torch torchvision matplotlib tqdm
