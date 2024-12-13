## Project Overview
# Summary of Project
A simple CNN model for animal classification with animal images using PyTorch
Goal: Classify animal species from images (zebra, elephant, giraffe, lion, hippo).
Approach: Shifted from animal crowds to single-animal images for better results.
Outcome: Achieved 90% validation accuracy and 80% test accuracy with SimpleCNN.
Challenges
Animal Crowds:
Issues with background noise, varying art styles, and partial animal bodies.
Models struggled with insufficient data and complex images.
Complex Models (VGG16):
Underfitting with small datasets and large model architectures.

## Quick Review of Results
Demo/Final_Project.ipynb
Demo/Final_project.pdf

# Data preparation
Animal Crowd Images
DALL-E-3 1024x1024 100imgs 

Single Animal Images
DALL-E-2 256x256 500 imgs

Real World Animal Images
13 imgs, just for quick verfication of the model's generalization


# Setup Instructions
## Requirements
- Python 3.12+
- PyTorch 2.0+

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

## How to Run
There are three main params.
should_train_model = True # Control whether to train the model again or use saved best model checkpoint
should_plot_results_grid = True # Control whether to plot the results grid, not valid for real animal images
use_real_animal_images = False # if True, will use real animal images for test validation

If you want to use saved best model checkpoint to test on real data, follow below settings.
should_train_model = False
should_plot_results_grid = False # MUST!
use_real_animal_images = True
You can also add more real animal images into data/realAnimalPics folder but it must be RGB image in .png format.

## Expected Output

# Training Progress Validation Results with plot
Training Epochs:   0%|                                         | 0/10 [00:00<?, ?it/s]Epoch 1/10, Training Loss: 7.2397, Validation Accuracy: 88.00%
Saved best model with accuracy: 88.00%
Training Epochs:  10%|███▎                             | 1/10 [00:02<00:25,  2.89s/itEpoch 2/10, Training Loss: 0.2647, Validation Accuracy: 92.00%
Saved best model with accuracy: 92.00%
Training Epochs:  20%|██████▌                          | 2/10 [00:05<00:23,  2.88s/it]Epoch 3/10, Training Loss: 0.0878, Validation Accuracy: 94.00%
Saved best model with accuracy: 94.00%
Training Epochs:  30%|█████████▉                       | 3/10 [00:08<00:20,  2.90s/it]Epoch 4/10, Training Loss: 0.0179, Validation Accuracy: 96.00%
Saved best model with accuracy: 96.00%
Training Epochs:  40%|█████████████▏                   | 4/10 [00:11<00:16,  2.73s/it]Epoch 5/10, Training Loss: 0.0065, Validation Accuracy: 92.00%
Training Epochs:  50%|████████████████▌                | 5/10 [00:13<00:13,  2.64s/it]Epoch 6/10, Training Loss: 0.0020, Validation Accuracy: 92.00%
Training Epochs:  60%|███████████████████▊             | 6/10 [00:16<00:10,  2.57s/it]Epoch 7/10, Training Loss: 0.0011, Validation Accuracy: 92.00%
Training Epochs:  70%|███████████████████████          | 7/10 [00:18<00:07,  2.53s/it]Epoch 8/10, Training Loss: 0.0008, Validation Accuracy: 92.00%
Training Epochs:  80%|██████████████████████████▍      | 8/10 [00:21<00:05,  2.58s/it]Epoch 9/10, Training Loss: 0.0006, Validation Accuracy: 92.00%
Training Epochs:  90%|█████████████████████████████▋   | 9/10 [00:23<00:02,  2.52s/it]Epoch 10/10, Training Loss: 0.0005, Validation Accuracy: 92.00%
Training Epochs: 100%|████████████████████████████████| 10/10 [00:26<00:00,  2.61s/it]
Plot saved to results/training_progress.png
![Alt text](results/training_progress.png)

# Test Results with plot
--- Evaluation Results ---
Test Accuracy: 92.00%

Sorted Classification Report by Precision:

              precision    recall  f1-score  support
zebra          1.000000  1.000000  1.000000      8.0
giraffe        1.000000  1.000000  1.000000     10.0
lion           1.000000  0.866667  0.928571     15.0
weighted avg   0.932000  0.920000  0.923016     50.0
macro avg      0.917778  0.926111  0.919048     50.0
hippo          0.888889  0.888889  0.888889      9.0
elephant       0.700000  0.875000  0.777778      8.0
![Alt text](results/test_results.png)


## Pre-trained Model Link
Not applicable, Both AnimalSpeciesCNN and VGG16 are underfitting and not adopted for this task


