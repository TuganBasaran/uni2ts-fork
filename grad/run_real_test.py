import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/tugan_basaran/Desktop/Lessons/CS.401/Kodlar/Moirai')

# Custom modules
from utils.TSFMTrainer import TSFMTrainer
from TSFM import TSFM

torch.manual_seed(42)


def calculate_regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

def calculate_financial_metrics(y_true, y_pred):
    # y_true and y_pred are arrays of targets.
    # Because they are sequential, y_true[i] is price at t+1 if y_true[i-1] is price at t.
    # Therefore, the price at t is y_true[:-1] and price at t+1 is y_true[1:]
    
    # We drop the first prediction from accuracy tracking because we don't have its y_{t-1} in this array.
    price_t = y_true[:-1]
    price_t1_true = y_true[1:]
    price_t1_pred = y_pred[1:]
    
    actual_dir = np.sign(price_t1_true - price_t)
    pred_dir = np.sign(price_t1_pred - price_t)
    
    # Handle zeros: if no actual move, we don't punish if prediction is also 0, though highly unlikely real prices are exactly 0 move.
    hit_ratio = np.mean(actual_dir == pred_dir)
    
    # Returns
    actual_returns = (price_t1_true - price_t) / price_t
    
    # Strategy Return
    # If pred_dir > 0, long (+1)
    # If pred_dir < 0, short (-1)
    # If pred_dir == 0, hold (0)
    strat_returns = pred_dir * actual_returns
    
    cum_strat_returns = np.cumprod(1 + strat_returns)
    cum_bh_returns = np.cumprod(1 + actual_returns)
    
    total_strat_ret = cum_strat_returns[-1] - 1
    total_bh_ret = cum_bh_returns[-1] - 1
    
    # Sharpe Ratio (annualized over 252 trading days)
    mean_ret = np.mean(strat_returns)
    std_ret = np.std(strat_returns)
    sharpe = np.sqrt(252) * (mean_ret / std_ret) if std_ret > 0 else 0
    
    # Max Drawdown
    drawdowns = cum_strat_returns / np.maximum.accumulate(cum_strat_returns) - 1
    max_dd = np.min(drawdowns)
    
    return {
        "Hit Ratio": hit_ratio,
        "Strategy Cumulative Return": total_strat_ret,
        "Buy & Hold Return": total_bh_ret,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }


def main():
    print("Loading Moirai Embeddings...")
    emb_path = '/Users/tugan_basaran/Desktop/Lessons/CS.401/Kodlar/Moirai/uni2ts/grad/moirai_embeddings.pt'
    data_path = '/Users/tugan_basaran/Desktop/Lessons/CS.401/Kodlar/Moirai/uni2ts/data/commodity_prices.csv'
    
    try:
        embedding_dict = torch.load(emb_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading {emb_path}: {e}")
        return

    print("Initializing TSFMTrainer...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    trainer = TSFMTrainer(device=device)

    
    # seq_len=32 is derived as context_length // patch_len
    # 512 // 16 = 32
    target_col = "CO1_Commodity"
    print(f"Preparing data for {target_col} target...")
    train_loader, val_loader, test_loader = trainer.prepare_data(
        embedding_dict=embedding_dict,
        data_path=data_path,
        target_col=target_col,
        index_col="date",
        context_length=512,
        patch_len=16,
        corr_period=30,
        threshold=0.3,
        forecast_horizon=1,
        batch_size=32
    )

    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("Starting Training...")
    epochs = 15
    history = trainer.fit(train_loader, val_loader, optimizer, loss_fn, epochs=epochs)

    print("Plotting Loss History...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), history["train_loss"], label="Train Loss", marker='o')
    plt.plot(range(1, epochs+1), history["val_loss"], label="Validation Loss", marker='s')
    plt.title(f"TSFM Training vs Validation Loss (Target: {target_col})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(data_path), "..", "loss_plot.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

    print("Evaluating on Test Set...")
    trainer.model.eval()
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            targets = batch.y
            node_preds = trainer.model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Sadece hedef düğümün tahminlerini alalım
            batch_size_current = batch.ptr.size(0) - 1
            preds = []
            target_idx_tensor = batch.target_idx
            
            for i in range(batch_size_current):
                start = batch.ptr[i]
                local_target = target_idx_tensor[i].item()
                pred = node_preds[start + local_target]
                preds.append(pred)
                
            preds = torch.stack(preds).squeeze()
            
            # CPU ve numpy formatına al
            y_true_list.extend(targets.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    print("\n--- Regression Metrics on Test Data ---")
    reg_metrics = calculate_regression_metrics(y_true, y_pred)
    for k, v in reg_metrics.items():
        print(f"{k}: {v:,.4f}")

    print("\n--- Financial Metrics on Test Data ---")
    fin_metrics = calculate_financial_metrics(y_true, y_pred)
    for k, v in fin_metrics.items():
        if "Ratio" in k or "Drawdown" in k:
            print(f"{k}: {v:,.4f}")
        else:
            print(f"{k}: {v:,.2%}")
            
    print("\n✅ Real test suite completed successfully!")

if __name__ == "__main__":
    main()
