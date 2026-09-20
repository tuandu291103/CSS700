"""Bounded CPU CNN training with no test-data argument or test evaluation path."""

import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from css700.models import SmallTabularImageCNN
from css700.training.grid_cnn import PredictionResult, _loader, evaluate_classifier


@dataclass(frozen=True)
class ValidationTrainingOutcome:
    model: SmallTabularImageCNN
    history: list[dict]
    best_epoch: int
    validation: PredictionResult
    optimizer_steps: int
    stop_reason: str
    parameters_changed: bool


def train_validation_cnn(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    max_seconds: float,
    on_epoch: Callable | None = None,
) -> ValidationTrainingOutcome:
    """Row-weighted training and validation macro-F1 selection, earliest tie retained."""
    if min(epochs, batch_size, patience, max_seconds, learning_rate) <= 0 or weight_decay < 0:
        raise ValueError("invalid training budget")
    for images, labels in ((train_images, train_labels), (validation_images, validation_labels)):
        if (
            images.ndim != 4
            or not len(images)
            or len(images) != len(labels)
            or images.dtype != np.float32
            or not np.isfinite(images).all()
            or labels.dtype != np.int64
            or labels.ndim != 1
            or not np.array_equal(np.unique(labels), [0, 1])
        ):
            raise ValueError("finite float32 images and both int64 binary classes required")
    if train_images.shape[1:] != validation_images.shape[1:]:
        raise ValueError("train/validation image shapes differ")
    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            model = SmallTabularImageCNN(input_channels=train_images.shape[1], num_classes=2).cpu()
            initial = deepcopy(model.state_dict())
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
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
            history, best_state = [], None
            best_score, best_epoch, stale, steps = -1.0, 0, 0, 0
            stop_reason = "epoch_limit"

            def check_time():
                if time.perf_counter() - started > max_seconds:
                    raise TimeoutError(
                        "fixed training wall-time limit exceeded; do not silently rerun"
                    )

            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0.0
                for inputs, labels in train_loader:
                    check_time()
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(inputs), labels)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    if any(
                        p.grad is None or not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    ):
                        raise ValueError("nonfinite or missing training gradient")
                    optimizer.step()
                    steps += 1
                    total_loss += float(loss.item()) * len(labels)
                validation = evaluate_classifier(
                    model, validation_loader, device=torch.device("cpu"), num_classes=2
                )
                if not np.isfinite(validation.logits).all() or not np.isfinite(validation.loss):
                    raise ValueError("nonfinite validation result")
                score = validation.metrics["macro_f1"]
                improved = score > best_score + 1e-12
                if improved:
                    best_score, best_epoch, best_state, stale = (
                        score,
                        epoch,
                        deepcopy(model.state_dict()),
                        0,
                    )
                else:
                    stale += 1
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": total_loss / len(train_labels),
                        "validation_loss": validation.loss,
                        "validation_accuracy": validation.metrics["accuracy"],
                        "validation_macro_f1": score,
                        "selected_new_best": improved,
                        "optimizer_steps": steps,
                    }
                )
                if on_epoch is not None:
                    on_epoch(history, model, improved)
                check_time()
                if stale >= patience:
                    stop_reason = "early_stopping_patience"
                    break
            if best_state is None:
                raise RuntimeError("no validation-selected checkpoint")
            model.load_state_dict(best_state)
            validation = evaluate_classifier(
                model, validation_loader, device=torch.device("cpu"), num_classes=2
            )
            check_time()
            return ValidationTrainingOutcome(
                model,
                history,
                best_epoch,
                validation,
                steps,
                stop_reason,
                any(not torch.equal(initial[k], v) for k, v in model.state_dict().items()),
            )
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism, warn_only=old_warn)
