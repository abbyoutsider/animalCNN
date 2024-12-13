# Animal Species Classification using CNN

## Project Overview
This project aims to classify images of animals into one of five predefined species using a Convolutional Neural Network (CNN). The species are:

- **Zebra**
- **Elephant**
- **Giraffe**
- **Lion**
- **Hippo**

The primary objective is to address challenges in image classification and develop a model that can accurately identify animal species from images with consistent quality.

## Real-World Applications
The classification of animal species has numerous practical applications, including:

- **Wildlife Monitoring**: Tracking and studying animal populations in their natural habitats.
- **Animal Conservation**: Supporting efforts to protect endangered species.
- **Automated Image Labeling**: Facilitating research and database management.

## Challenges in Image Classification
- **Variability in Appearance**: Animals of the same species can vary significantly in size, shape, and color.
- **Background Noise**: Natural images often include cluttered or distracting backgrounds.
- **Complex Environments**: Real-world images may contain multiple animals or incomplete views of the subject.

## Project Evolution

### Initial Approach: Crowded Animal Images
- **Dataset**: 100 images generated using DALL-E 3, featuring 1 to 5 animals with uncontrolled backgrounds and art styles.
- **Challenges**:
  - High cost and time for generating high-resolution images.
  - Inconsistent art styles and partial animal bodies.
  - Models (VGG11, VGG16) achieved less than 20% accuracy due to noise and insufficient data.

### New Approach: Single Animal Images
- **Dataset**: 256x256 images generated using DALL-E 2 with prompts like "A realistic photo of exactly ONE whole giraffe with a white background."
- **Benefits**:
  - Clearer features with single animals in the frame.
  - Consistent white backgrounds and art styles.
  - Improved image quality and composition.

### Models Tested
#### 1. **Pre-trained VGG16**
- **Architecture**: 16 convolutional layers, input size 224x224, and 5 pooling layers.
- **Result**: Validation accuracy below 20%, struggling with small datasets and underfitting.

#### 2. **Self-trained AnimalSpeciesCNN**
- **Architecture**: 3 convolutional layers, input size 1024x1024, and 3 pooling layers.
- **Result**: Validation accuracy below 20%, struggling with small datasets and underfitting.

#### 3. **Self-trained Simplified Model (SimpleCNN)**
- **Architecture**: Custom CNN with 1 convolutional layer, input size 256x256, and 1 pooling layer.
- **Performance**:
  - Validation accuracy: 86%-96% (peaking at epoch 3).
  - Test accuracy: 80%.
  - Generalized well to real animal images, achieving 61.54% accuracy on 13 real-world examples.

## Key Results
- **Validation Accuracy**: 90% with SimpleCNN.
- **Test Accuracy**: 80%.
- **Generalization**: Demonstrated effectiveness on real animal pictures, outperforming random guessing (20% accuracy).

### Metrics
- **Precision**: 0.93
- **Recall**: 0.92
- **F1-score**: 0.92

## Summary
- **Goal**: To classify animal species from images using CNN.
- **Approach**: Shifted from complex, noisy datasets of animal crowds to simpler, cleaner datasets with single animals and white backgrounds.
- **Outcome**: Successfully developed a simplified CNN model achieving high accuracy and demonstrating potential for real-world application.

## Future Work
- Expand the dataset to include more images per class.
- Apply data augmentation to improve model robustness.
- Explore semi-supervised learning methods to handle larger datasets.

## Repository Structure
```plaintext
├── data/                # Dataset directory
├── models/              # Saved models and training scripts
├── notebooks/           # Jupyter notebooks for experimentation
├── README.md            # Project overview
├── requirements.txt     # Dependencies
└── src/                 # Source code
    ├── train.py         # Training script
    ├── data_loader.py   # Load datasets and dataloader
    └── utils.py         # Utility functions
    └── model.py         # Define Model 
    └── main.py          # Main script
    


## Quick Review of Results
Demo/Final_Project.ipynb

Demo/Final_project.pdf

## Data preparation
Animal Crowd Images : DALL-E-3 1024x1024 100imgs 

Single Animal Images: DALL-E-2 256x256 500 imgs

Real World Animal Images
13 imgs, just for quick verfication of the model's generalization


## Setup Instructions
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

Training Progress Validation Results with plot

![Alt text](<Screenshot 2024-12-14 at 5.53.28 AM.png>)
![Alt text](results/training_progress.png)

Test Results with plot
![Alt text](<Screenshot 2024-12-14 at 6.47.26 AM.png>)
![Alt text](results/test_results.png)


Pre-trained Model Link
Not applicable, Both AnimalSpeciesCNN and VGG16 are underfitting and not adopted for this task


