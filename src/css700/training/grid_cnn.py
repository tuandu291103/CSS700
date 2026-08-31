"""Deterministic training and evaluation for the Grid CNN pilot."""

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from css700.models import SmallTabularImageCNN


@dataclass(frozen=True)
class PredictionResult:
    """Predictions and metrics from one frozen model and split."""

    loss: float
    logits: NDArray[np.float32]
    probabilities: NDArray[np.float32]
    predictions: NDArray[np.int64]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class TrainingOutcome:
    """Best validation-selected model plus final validation and test results."""

    model: SmallTabularImageCNN
    history: tuple[dict[str, float | int], ...]
    best_epoch: int
    validation: PredictionResult
    test: PredictionResult
    majority_class: int
    majority_metrics: dict[str, Any]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _classification_metrics(
    labels: NDArray[np.int64], predictions: NDArray[np.int64], *, num_classes: int
) -> dict[str, Any]:
    class_labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "per_class_precision": precision_score(
            labels, predictions, labels=class_labels, average=None, zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            labels, predictions, labels=class_labels, average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_labels).tolist(),
    }


def _loader(
    images: NDArray[np.float32],
    labels: NDArray[np.int64],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
) -> PredictionResult:
    """Evaluate without gradients and retain sample-ordered model outputs."""

    criterion = nn.CrossEntropyLoss(reduction="sum")
    logits_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            total_loss += float(criterion(logits, labels).item())
            logits_parts.append(logits.cpu())
            labels_parts.append(labels.cpu())

    logits_tensor = torch.cat(logits_parts)
    labels_array = torch.cat(labels_parts).numpy().astype(np.int64, copy=False)
    probabilities = torch.softmax(logits_tensor, dim=1).numpy().astype(np.float32, copy=False)
    predictions = probabilities.argmax(axis=1).astype(np.int64, copy=False)
    return PredictionResult(
        loss=total_loss / len(labels_array),
        logits=logits_tensor.numpy().astype(np.float32, copy=False),
        probabilities=probabilities,
        predictions=predictions,
        metrics=_classification_metrics(labels_array, predictions, num_classes=num_classes),
    )


def train_grid_cnn(
    train_images: NDArray[np.float32],
    train_labels: NDArray[np.int64],
    validation_images: NDArray[np.float32],
    validation_labels: NDArray[np.int64],
    test_images: NDArray[np.float32],
    test_labels: NDArray[np.int64],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    early_stopping_patience: int,
    weight_decay: float = 0.0,
    device_name: str = "cpu",
) -> TrainingOutcome:
    """Train on training data, select on validation macro F1, then evaluate test once."""

    if epochs <= 0 or batch_size <= 0 or early_stopping_patience <= 0:
        raise ValueError("epochs, batch_size, and early_stopping_patience must be positive")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    _seed_everything(seed)

    num_classes = int(np.unique(train_labels).size)
    model = SmallTabularImageCNN(input_channels=train_images.shape[1], num_classes=num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = _loader(
        train_images, train_labels, batch_size=batch_size, shuffle=True, seed=seed
    )
    validation_loader = _loader(
        validation_images,
        validation_labels,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = _loader(test_images, test_labels, batch_size=batch_size, shuffle=False, seed=seed)

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item()) * len(labels)

        validation = evaluate_classifier(
            model, validation_loader, device=device, num_classes=num_classes
        )
        validation_score = float(validation.metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_train_loss / len(train_labels),
                "validation_loss": validation.loss,
                "validation_accuracy": float(validation.metrics["accuracy"]),
                "validation_macro_f1": validation_score,
            }
        )
        if validation_score > best_score + 1e-12:
            best_score = validation_score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    validation = evaluate_classifier(
        model, validation_loader, device=device, num_classes=num_classes
    )
    test = evaluate_classifier(model, test_loader, device=device, num_classes=num_classes)

    majority_class = int(np.bincount(train_labels).argmax())
    majority_predictions = np.full_like(test_labels, majority_class)
    majority_metrics = _classification_metrics(
        test_labels, majority_predictions, num_classes=num_classes
    )
    return TrainingOutcome(
        model=model.cpu(),
        history=tuple(history),
        best_epoch=best_epoch,
        validation=validation,
        test=test,
        majority_class=majority_class,
        majority_metrics=majority_metrics,
    )
