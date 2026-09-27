"""Training and TorchMetrics evaluation helpers adapted from the notebook."""

import torch


def evaluate_tm(model, data_loader, metric):
    """Evaluate a TorchMetrics metric over a complete data loader."""
    device = next(model.parameters()).device
    metric = metric.to(device)
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()


def train2(model, optimizer, criterion, metric, train_loader, valid_loader,
           n_epochs):
    """Train and return per-epoch loss, training metric, and validation metric.

    Pass a TorchMetrics multiclass accuracy metric with num_classes=10 for
    FashionMNIST. History keys match the notebook: train_losses,
    train_metrics (training accuracy), and valid_metrics (validation accuracy).
    Loss is averaged over batches, as in the notebook.
    """
    device = next(model.parameters()).device
    metric = metric.to(device)
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        metric.reset()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.update(y_pred.detach(), y_batch)

        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric).item()
        )
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history
