import torch
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report

def train_model(model, train_loader, val_loader, optimizer, loss_func, best_model_path, epochs, device):
    best_accuracy = 0.0
    train_loss_list = []  # List to store training loss per epoch
    validation_accuracy_list = []  # List to store validation accuracy per epoch
    mini_batch_loss_list = []  # List to store loss for each mini-batch

    for epoch in tqdm(range(epochs), desc="Training Epochs"):  # Loop over epochs
        model.train()
        running_loss = 0.0  # Variable to accumulate loss for this epoch
        
        # For each mini-batch, grab the data and perform forward/backward pass
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # Accumulate loss for this mini-batch
            mini_batch_loss_list.append(loss.item())  # Store mini-batch loss

        avg_train_loss = running_loss / len(train_loader)  # Average loss for the epoch
        train_loss_list.append(avg_train_loss)  # Store the average loss per epoch
        
        # Validate after every epoch
        val_accuracy = validate_model(model, val_loader, device)
        validation_accuracy_list.append(val_accuracy)  # Store validation accuracy
        
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Check if the validation accuracy has improved and save the model if it has
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with accuracy: {best_accuracy:.2f}%")

    return train_loss_list, validation_accuracy_list, mini_batch_loss_list  # Return the lists

def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    return 100 * correct / total


def evaluate_model(model, best_model_path, test_loader, device, class_names):
    """
    Evaluate the model using the best checkpoint and print detailed results.

    Args:
    - model: The trained model.
    - best_model_path: Path to the best checkpoint file.
    - test_loader: DataLoader containing the test dataset.
    - device: Device to run the model on (CPU or GPU).
    - class_names: List of class names corresponding to integer labels.
    """
    # Load the best model checkpoint
    model.load_state_dict(torch.load(best_model_path, weights_only=True))  # Load weights safely
    model.eval()  # Set model to evaluation mode
    model.to(device)  # Move model to the specified device
    
    # Initialize variables to store predictions and targets
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # Forward pass
            predictions = torch.argmax(outputs, dim=1)  # Get predicted class
            all_predictions.extend(predictions.cpu().numpy())  # Collect predictions
            all_targets.extend(targets.cpu().numpy())  # Collect true labels

    # Calculate overall test accuracy
    total_correct = sum([pred == target for pred, target in zip(all_predictions, all_targets)])
    total_samples = len(all_targets)
    test_accuracy = 100 * total_correct / total_samples

    # Generate a classification report as a dictionary
    report_dict = classification_report(
        all_targets, all_predictions, target_names=class_names, output_dict=True
    )

    # Convert the dictionary to a DataFrame for sorting
    report_df = pd.DataFrame(report_dict).transpose()

    # Exclude the "accuracy" row and sort by precision
    sorted_report_df = report_df.drop(["accuracy"], errors="ignore").sort_values(
        by="precision", ascending=False
    )

    # Print results
    print("\n--- Evaluation Results ---")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("\nSorted Classification Report by Precision:\n")
    print(sorted_report_df)

    return all_predictions, all_targets, test_accuracy, sorted_report_df