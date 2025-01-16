# Music Restoration

## Overview
This project provides a framework for audio denoising and generation using machine learning models. The core functionality includes:
- Loading and preprocessing audio datasets.
- Training, validating, and testing models for audio denoising and generation.
- Evaluation of model performance.

## File Structure

### 1. `dataloader.py`
This module handles loading and preprocessing audio datasets. Key features include:
- Loading audio files and their metadata.
- Applying transformations such as spectrogram generation, noise addition, and silence insertion.
- Splitting datasets into training, validation, and test sets.
- Creating PyTorch dataloaders.

### 2. `validation.py`
Contains utilities for evaluating model performance, including:
- Metrics computation (e.g., MSE).
- Functions for inverse spectrogram transformations.
- Integration with pretrained models for evaluation.

### 3. `train.py`
Handles the training process of the models. It includes:
- Training steps for denoiser and GAN components.
- Epoch-wise logging of training and validation metrics.
- Saving trained models for further evaluation or deployment.

### 4. `model.py`
Defines the neural network architectures used in the project:
- **Denoising Autoencoder**: For noise reduction in audio signals.
- **U-Net Generator**: For enhancing audio quality.
- **Discriminator**: For adversarial training in GANs.

### 5. `loss.py`
Provides loss functions for training, including:
- Mean Squared Error (MSE) for evaluating the difference between clean and denoised/generated audio.


## Usage Instructions

### 1. Dataset Preparation
Ensure you have the FMA (Free Music Archive) dataset downloaded and organized as follows:
```
base_dir/
    fma_small/
        000/
        001/
        ...
    fma_metadata/
        tracks.csv
```

### 2. Setting Up the Environment
- Install the required Python libraries:
  ```bash
  pip install torch torchaudio librosa pandas
  ```

### 3. Running the Code

#### Training
To train the models, run:
```bash
python train.py
```

### 4. Model Evaluation
The validation script generates metrics like total MSE, denoiser MSE, and generator MSE. These metrics can be used to compare model performance.

### 5. Pretrained Models
Pretrained models can be loaded using the paths specified in the scripts:
- `denoiser_wave_final.pt`
- `generator_wave_final.pt`
- `discriminator_wave_final.pt`

