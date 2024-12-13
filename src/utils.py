import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image

def normalize_images(image_files, image_dir, should_resize):
    normalized_images = []
    for img_path in image_files:
        img_full_path = os.path.join(image_dir, img_path)
        # img = cv2.imread(img_full_path)
        img = Image.open(img_full_path)
        if img is None:
            print(f"Failed to load image: {img_full_path}")
        else:
            img = np.array(img) / 255.0
        if should_resize:
            # Resize the image to 256x256
            img = cv2.resize(img, (256, 256))
        img = np.transpose(img, (2, 0, 1))
        normalized_images.append(img)
    return np.array(normalized_images)

def plot_images(image_files, predictions, ground_truths, animal_list, image_dir, save_path):
    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(20, 10))
    axes = axes.flatten()
    for i, img_path in enumerate(image_files):
        img_full_path = os.path.join(image_dir, img_path)
        img = plt.imread(img_full_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        predicted_label = animal_list[predictions[i]]
        ground_truth_label = animal_list[ground_truths[i]]
        axes[i].set_title(f"P: {predicted_label}\nG: {ground_truth_label}", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_training_progress(train_loss_list, validation_accuracy_list, mini_batch_loss_list, save_path):
    """
    Function to plot training loss and validation accuracy, and save the plot as a PNG.

    Args:
    - train_loss_list: List of training losses recorded during training.
    - validation_accuracy_list: List of validation accuracies recorded during training.
    - mini_batch_loss_list: List of training losses per mini-batch.
    - save_path: Path where the plot will be saved.
    """
    sns.set(style='whitegrid', font_scale=1)

    # Create a figure with three subplots (3 rows and 1 column)
    plt.figure(figsize=(15, 12))  # Adjust the size to fit three subplots

    # Plot training loss per epoch
    plt.subplot(3, 1, 1)
    plt.plot(train_loss_list, linewidth=3, color='blue')
    plt.ylabel("Training Loss per Epoch")
    plt.xlabel("Epochs")
    sns.despine()

    # Plot training loss per mini-batch
    plt.subplot(3, 1, 2)
    plt.plot(mini_batch_loss_list, linewidth=3, color='orange')
    plt.ylabel("Training Loss per Mini-Batch")
    plt.xlabel("Iterations")
    sns.despine()

    # Plot validation accuracy
    plt.subplot(3, 1, 3)
    plt.plot(validation_accuracy_list, linewidth=3, color='green')
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Epochs")
    sns.despine()

    # Save the plot as a PNG file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist
    plt.tight_layout()
    plt.savefig(save_path)  # Save the figure to the given path
    print(f"Plot saved to {save_path}")

    # Display the plot
    plt.show()

