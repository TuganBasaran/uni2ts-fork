import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "/Users/tugan_basaran/Desktop/Lessons/CS.401/Kodlar/Moirai")

# Custom modules
from utils.TSFMTrainer import TSFMTrainer
from TSFM import TSFM


def calculate_regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def calculate_financial_metrics(y_true, y_pred):
    price_t = y_true[:-1]
    price_t1_true = y_true[1:]
    price_t1_pred = y_pred[1:]

    actual_dir = np.sign(price_t1_true - price_t)
    pred_dir = np.sign(price_t1_pred - price_t)

    hit_ratio = np.mean(actual_dir == pred_dir)

    actual_returns = (price_t1_true - price_t) / price_t
    strat_returns = pred_dir * actual_returns

    cum_strat_returns = np.cumprod(1 + strat_returns)
    cum_bh_returns = np.cumprod(1 + actual_returns)

    total_strat_ret = cum_strat_returns[-1] - 1
    total_bh_ret = cum_bh_returns[-1] - 1

    mean_ret = np.mean(strat_returns)
    std_ret = np.std(strat_returns)
    sharpe = np.sqrt(252) * (mean_ret / std_ret) if std_ret > 0 else 0

    drawdowns = cum_strat_returns / np.maximum.accumulate(cum_strat_returns) - 1
    max_dd = np.min(drawdowns)

    return {
        "Hit Ratio": hit_ratio,
        "Strategy Cumulative Return": total_strat_ret,
        "Buy & Hold Return": total_bh_ret,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }


def main():
    print("Loading Moirai Embeddings...")
    emb_path = "/Users/tugan_basaran/Desktop/Lessons/CS.401/Kodlar/Moirai/uni2ts/Embeddings/ten_comdty_embeddings.pt"
    data_path = "/Users/tugan_basaran/Desktop/Lessons/CS.401/Kodlar/Moirai/uni2ts/data/first_ten_columns.csv"

    try:
        raw_emb_dict = torch.load(emb_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading {emb_path}: {e}")
        return

    # In this new dataset, column names match the embedding keys natively (CO1 Comdty, CL1 Comdty, etc.)
    # No need to map to Yahoo Tickers (BZ=F).
    embedding_dict = raw_emb_dict

    print("Initializing TSFMTrainer...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = TSFMTrainer(device=device)

    print("Preparing data for CO1 Comdty target...")
    target_col = "CO1 Comdty"
    train_loader, val_loader, test_loader = trainer.prepare_data(
        embedding_dict=embedding_dict,
        data_path=data_path,
        target_column=target_col,
        index_col="date",
        batch_size=32,
    )

    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss()  # More robust to outliers in price movements

    print("Starting Training...")
    epochs = 15
    trainer.train(
        loss_func=loss_fn, optimizer=optimizer, epochs=epochs, learning_rate=1e-3
    )

    print("Evaluating on Test Set...")
    trainer.model.eval()
    y_true_list = []
    y_pred_list = []
    price_t_list = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            targets = batch.y

            preds = trainer.model(
                e_i=batch.e_i,
                x=batch.x,
                edge_index=batch.edge_index,
                edge_weight=batch.edge_attr,
                target_idx=batch.target_idx,
                batch_ptr=batch.ptr,
            )

            y_true_list.extend(targets.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
            price_t_list.extend(batch.price_t.cpu().numpy())

    y_true_ret = np.array(y_true_list).reshape(-1, 1)
    y_pred_ret = np.array(y_pred_list).reshape(-1, 1)
    price_t = np.array(price_t_list).flatten()

    # Inverse transform to get actual raw percentage returns
    y_true_ret = trainer.target_scaler.inverse_transform(y_true_ret).flatten()
    y_pred_ret = trainer.target_scaler.inverse_transform(y_pred_ret).flatten()

    # Reconstruct the Absolute Target Prices!
    # Price_{t+1} = Price_t * (1 + Return_{t->t+1})
    y_true = price_t * (1 + y_true_ret)
    y_pred = price_t * (1 + y_pred_ret)

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

    print("\n✅ Real test suite completed successfully on first_ten_columns!")


if __name__ == "__main__":
    main()
