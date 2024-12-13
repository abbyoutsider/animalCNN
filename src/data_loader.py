import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils import normalize_images

# Load all image files from the image directory and return the normalized images and labels
def load_data(image_dir, specie2tag, should_resize):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    animal_species = [file.split('_')[0] for file in image_files]
    species_tags = [specie2tag[species] for species in animal_species]

    X = normalize_images(image_files, image_dir, should_resize)
    y = np.array(species_tags)
    return X, y, image_files

def split_data(X, y, should_plot_results_grid, test_size=0.2, val_size=0.5, random_state=42):
    if should_plot_results_grid: # By default is False
        # Split the dataset into training, validation and testing set
        # Index order as the data generation is already randomized, order for generate the test result grid
        X_train = X[:400]
        y_train = y[:400]
        X_val = X[400:450]
        y_val = y[400:450]
        X_test = X[450:]
        y_test = y[450:]
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
