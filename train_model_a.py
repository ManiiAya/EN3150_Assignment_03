import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay

from data_prep import get_dataloaders
from model_A import ModelA
from utils import count_parameters, train_model, evaluate_model

parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd", "sgd_momentum"])
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

# Default learning rates: Adam likes small, plain SGD needs larger
default_lr = {"adam": 1e-3, "sgd": 1e-2, "sgd_momentum": 1e-2}
lr = args.lr if args.lr is not None else default_lr[args.optimizer]

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | optimizer: {args.optimizer} | lr: {lr}")

train_loader, val_loader, test_loader, classes = get_dataloaders()

model = ModelA(num_classes=len(classes)).to(device)
print("Trainable parameters:", count_parameters(model))

criterion = nn.CrossEntropyLoss()
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

history = train_model(model, train_loader, val_loader, criterion, optimizer,
                      device, num_epochs=args.epochs)
results = evaluate_model(model, test_loader, device)

# Model size on disk (KB)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
tag = f"model_a_{args.optimizer}"
ckpt = f"checkpoints/{tag}.pt"
torch.save(model.state_dict(), ckpt)
size_kb = os.path.getsize(ckpt) / 1024

summary = {
    "optimizer": args.optimizer, "lr": lr, "epochs": args.epochs,
    "parameters": count_parameters(model),
    "model_size_kb": round(size_kb, 1),
    "avg_epoch_time_s": round(sum(history["epoch_time"][1:]) / (len(history["epoch_time"]) - 1), 2),
    "test_accuracy": results["accuracy"],
    "test_precision_macro": results["precision"],
    "test_recall_macro": results["recall"],
    "history": history,
}
with open(f"results/{tag}.json", "w") as f:
    json.dump(summary, f, indent=2)

# Loss curves
epochs = range(1, len(history["train_loss"]) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_loss"], label="Training loss")
plt.plot(epochs, history["val_loss"], label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
plt.title(f"Model A ({args.optimizer}) - loss")
plt.tight_layout()
plt.savefig(f"results/figures/{tag}_loss.png", dpi=150)
plt.close()

# Confusion matrix
fig, ax = plt.subplots(figsize=(9, 9))
ConfusionMatrixDisplay(results["confusion_matrix"], display_labels=classes).plot(
    ax=ax, xticks_rotation=45, values_format="d")
ax.set_title(f"Model A ({args.optimizer}) - confusion matrix")
plt.tight_layout()
plt.savefig(f"results/figures/{tag}_confusion.png", dpi=150)
plt.close()

print(f"\nTest acc: {results['accuracy']:.4f} | precision: {results['precision']:.4f} "
      f"| recall: {results['recall']:.4f}")
print(f"Model size: {size_kb:.1f} KB | avg epoch time: {summary['avg_epoch_time_s']} s")