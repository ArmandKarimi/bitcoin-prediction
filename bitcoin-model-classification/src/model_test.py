import torch
import numpy as np


def evaluate(model, loader, device):
    """
    Evaluates the model on the given data loader and returns the predictions
    and ground truth values.

    Parameters:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): The DataLoader for evaluation data.
        device (torch.device): The device to run the computations on.
        data_mean (dict or pd.Series): The mean values used for normalization (for 'Close').
        data_std (dict or pd.Series): The std values used for normalization (for 'Close').
        classification (bool): Whether the model is performing classification (True) or regression (False).

    Returns:
        tuple: (predictions, truths) as NumPy arrays.
    """
    model.eval()
    predictions, truths = [], []
    
    with torch.no_grad():
        for X, y in loader:
            # Move data to the correct device
            X, y = X.to(device), y.to(device)
            # Get model predictions and move to CPU as NumPy array
            preds = model(X).cpu().numpy()

            preds = (torch.sigmoid(torch.tensor(preds)) > 0.5).float().numpy()  # Convert logits to binary
            truths.extend(y.cpu().numpy())  # Keep y unchanged (binary 0/1)

            predictions.extend(preds)

    return np.array(predictions), np.array(truths)
