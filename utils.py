import time
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)


def count_parameters(model):
    """Return the number of trainable parameters in a model."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def model_size_kb(model):
    """Return the size of the model in kilobytes."""
    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total_params * 4 / 1024


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train a model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate statistics
        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def validate(model, loader, criterion, device):
    """Evaluate loss and accuracy on the validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    """Train a model and record training/validation history."""
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_time": [],
    }
    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        epoch_time = time.perf_counter() - start_time
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["epoch_time"].append(epoch_time)
        print(
            f"Epoch [{epoch + 1:02d}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )
    return history


def evaluate_model(model, loader, device):
    """Evaluate a trained model on a dataset."""
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    recall = recall_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(true_labels, predicted_labels)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
    }


def plot_training_history(history, model_name="Model"):
    """Plot training and validation loss and accuracy."""
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, model_name="Model"):
    """Plot a row-normalized confusion matrix."""
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    fig, ax = plt.subplots(figsize=(9, 9))
    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format=".2f",
        cmap="Blues",
        normalize="true",
    )
    ax.set_title(f"{model_name} - Normalized Confusion Matrix")
    plt.tight_layout()
    plt.show()