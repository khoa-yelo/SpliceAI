import argparse
import logging
import os
import sys

import torch
import torch.functional as F
from torch.utils.data import DataLoader

import numpy as np
from collections import Counter

from dataloader import SpliceDataset
from model import SpliceAI_10k, SpliceAI_2k, SpliceAI_400nt, SpliceAI_80nt
from metrics import topk_accuracy, pr_auc

def parse_args():
    parser = argparse.ArgumentParser(description="Train SpliceAI model")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of training data to use")
    parser.add_argument("--val_fraction", type=float, default=1.0, help="Fraction of validation data to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--flank", type=int, default=5000, help="Flank size for training data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model checkpoints")
    return parser.parse_args()

def get_dataloader(args):
     # Create datasets
    train_dataset = SpliceDataset(args.train_data, args.train_fraction, flank=args.flank)
    val_dataset = SpliceDataset(args.val_data, args.val_fraction, flank=args.flank)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1
    )
    return train_loader, val_loader, train_dataset, val_dataset

def get_model(args):
    if args.flank == 5000:
        model = SpliceAI_10k(in_channels=4, mid_channels=32, out_channels=3)
    elif args.flank == 2000:
        model = SpliceAI_2k(in_channels=4, mid_channels=32, out_channels=3)
    elif args.flank == 400:
        model = SpliceAI_400nt(in_channels=4, mid_channels=32, out_channels=3)
    elif args.flank == 80:
        model = SpliceAI_80nt(in_channels=4, mid_channels=32, out_channels=3)
    else:
        raise ValueError("Invalid flank size. Choose from [5000, 2000, 400, 80]")
    return model


def compute_class_weights(dataset, num_classes):
    counts = Counter()
    total = 0
    for _, targets in dataset:
        labels = targets.reshape(-1).numpy()
        counts.update(labels)
        total += len(labels)
    freqs = [counts[i] / total for i in range(num_classes)]
    weights = [1.0 / f if f > 0 else 0.0 for f in freqs]
    weight_tensor = torch.tensor(weights)
    weight_tensor = weight_tensor / weight_tensor.sum()
    return weight_tensor

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    context = torch.enable_grad() if train else torch.no_grad()
    total_loss = 0.0
    all_logits, all_targets = [], []
    with context:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if train:
                optimizer.zero_grad()

            outputs = model(inputs)             # (B, C, L)
            B, C, L = outputs.shape
            outputs = outputs.permute(0, 2, 1).reshape(-1, C)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_logits.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    targets_onehot = F.one_hot(torch.tensor(targets), num_classes=probs.shape[1]).numpy()

    metrics = {
        "loss": total_loss / len(loader),
        "topk_acc": topk_accuracy(probs, targets_onehot),
        "pr_auc": pr_auc(probs, targets_onehot),
        "true_counts": np.bincount(targets, minlength=probs.shape[1]),
        "pred_counts": np.bincount(probs.argmax(axis=1), minlength=probs.shape[1]),
    }    
    return metrics

def train(args, logger):
    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(args)
    logger.info("Loaded datasets: %d training samples, %d validation samples", len(train_dataset), len(val_dataset))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args).to(device)
    class_weights = compute_class_weights(train_dataset, num_classes=3).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        logger.info("Epoch %d/%d", epoch, args.num_epochs)
        logger.info("  Train Loss: %.4f", train_metrics["loss"])
        logger.info("  Val   Loss: %.4f", val_metrics["loss"])

        for i, (ta, pa, ra, ca) in enumerate(zip(
                train_metrics["topk_acc"], val_metrics["topk_acc"],
                train_metrics["pr_auc"],   val_metrics["pr_auc"])):
            logger.info(
                "  Class %d: Top-k Acc → Train: %.3f, Val: %.3f;  "
                "PR-AUC → Train: %.3f, Val: %.3f",
                i, ta, pa, ra, ca
            )

        logger.info(
            "  Train counts (true → pred): %s → %s",
            train_metrics["true_counts"], train_metrics["pred_counts"]
        )
        logger.info(
            "  Val   counts (true → pred): %s → %s",
            val_metrics["true_counts"], val_metrics["pred_counts"]
        )
        logger.info("-" * 60)

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{args.output_dir}/model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("Model checkpoint saved to %s", checkpoint_path)
    logger.info("Training completed.")
    logger.info("Final model saved to %s", args.output_dir)
            
   

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler("spliceAI_train.log"), logging.StreamHandler(sys.stdout)],
                         format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args, logger)