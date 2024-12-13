import torch
from model import SimpleCNN
from data_loader import load_data, split_data
from train import train_model, evaluate_model
from torch.utils.data import DataLoader, TensorDataset
from utils import plot_training_progress, plot_images

"""
Predefined variables and paths:
"""
animal_list = ["zebra", "elephant", "giraffe", "lion", "hippo"]
specie2tag = {animal: i for i, animal in enumerate(animal_list)}
image_dir = "data/generatedAnimalPics"
real_image_dir = "data/realAnimalPics"
training_progress_save_path = "results/training_progress.png"
best_model_path = "checkpoints/best_model.pth"
test_result_save_path = "results/test_results.png"
should_train_model = True # Control whether to train the model again or use saved best model checkpoint
should_plot_results_grid = True # Only valid for [450:] fixed data. Control whether to plot the results grid, data split needs to be in order too
use_real_animal_images = False # if True, will need to resize the images to 256x256, by DALL-E-2

# Prepare datasets and dataloaders
X, y, image_files = load_data(image_dir, specie2tag, use_real_animal_images)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, should_plot_results_grid)
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=4)

# Try out will real world images
if use_real_animal_images:
    X_test, y_test, image_files = load_data(real_image_dir, specie2tag, use_real_animal_images)
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=4)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# Both AnimalSpeciesCNN and VGG16 are underfitting and not adopted for this task
# model = models.vgg16(pretrained=True)
# model.classifier[6] = nn.Linear(in_features=4096, out_features=5)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
loss_func = torch.nn.CrossEntropyLoss()

if should_train_model:
    train_loss_list, validation_accuracy_list, mini_batch_loss_list =  train_model(model, train_loader, val_loader, optimizer, loss_func, best_model_path, epochs=10, device=device)    

    # Plot the training progress
    plot_training_progress(train_loss_list, validation_accuracy_list, mini_batch_loss_list, training_progress_save_path)


all_predictions, ground_truths, test_accuracy, sorted_report_df = evaluate_model(model, best_model_path, test_loader, device, animal_list)

if should_plot_results_grid:
    test_image_files=image_files[450:]
    plot_images(test_image_files, all_predictions, ground_truths, animal_list, image_dir, test_result_save_path)
