"""Скрипт для обучения модели
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from src.data.data_loader import FaceMaskDataset
from src.models.baseline import BaselineModel

SRC_DIR = Path(__file__).resolve().parents[2]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_epoch(model, loader, optimizer, criterion, desc: str, train=True):
    mean_loss = 0.0
    y_pred, y_true = [], []
    epoch_progress = tqdm(
        enumerate(loader), desc=f"{desc} loss {mean_loss:.4}", total=len(loader)
    )
    for i, batch in epoch_progress:
        images: torch.Tensor = batch[0].to(device)
        labels: torch.Tensor = batch[1].to(device)

        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()

        mean_loss += float(loss.item())
        epoch_progress.set_description(f"{desc} loss {mean_loss / (i+1):.4}")
        y_true += labels.tolist()
        y_pred += outputs.argmax(dim=1).tolist()

    return mean_loss / len(loader), y_true, y_pred


def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", "-d", type=Path, help="Путь до датасетов")
    parser.add_argument("--epoch", "-e", type=int, help="Кол-во эпох для обучения")
    parser.add_argument("--batch-size", "-b", type=int, help="Размер батча")
    parser.add_argument("--lr", type=float, help="Learning rate")

    args = parser.parse_args()

    epoch: int = args.epoch
    batch_size: int = args.batch_size
    dataset_dir: Path = args.dataset_dir

    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    model = BaselineModel().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_dataset = FaceMaskDataset(dataset_dir / "Train", transform=transform)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = FaceMaskDataset(dataset_dir / "Test", transform=transform)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = FaceMaskDataset(dataset_dir / "Validation", transform=transform)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    train_logs = []
    valid_logs = []

    for e in range(epoch):
        train_loss, y_true, y_pred = make_epoch(
            model, train_data_loader, optimizer, criterion, f"#{e} Train", train=True
        )
        train_metrics = get_metrics(y_true, y_pred)
        train_metrics["loss"] = train_loss
        train_logs.append(train_metrics)
        valid_loss, y_true, y_pred = make_epoch(
            model, valid_data_loader, optimizer, criterion, f"#{e} Valid", train=False
        )
        valid_metrics = get_metrics(y_true, y_pred)
        valid_metrics["loss"] = valid_loss
        valid_logs.append(valid_metrics)

    _, y_true, y_pred = make_epoch(
        model, test_data_loader, optimizer, criterion, "Test", train=False
    )

    metrics = get_metrics(y_true, y_pred)
    with open(f"{SRC_DIR}/metrics/metrics.json", "w") as fid:
        json.dump(metrics, fid)

    torch.save(model.cpu(), SRC_DIR / "models/model.pkl")


if __name__ == "__main__":
    main()
