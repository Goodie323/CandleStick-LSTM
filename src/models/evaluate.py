import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, test_dataset, device="cuda"):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in test_dataset:
            X, y = X.to(device), y.to(device)
            pred = model(X.unsqueeze(0))  # single sample
            preds.append(pred.item())
            targets.append(y.item())
    
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    directional_acc = np.mean(
        (np.sign(np.diff(targets)) == np.sign(np.diff(preds))).astype(int)
    )
    
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Directional Accuracy: {directional_acc:.2%}")
    return rmse, mae, directional_acc
