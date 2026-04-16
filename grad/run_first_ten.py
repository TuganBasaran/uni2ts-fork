import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.DirectionalLoss import DirectionalLoss

sys.path.insert(0, "/Users/tugan_basaran/Desktop/Lessons/CS.401/Kodlar/Moirai")

# Custom modules
from utils.TSFMTrainer import TSFMTrainer
from TSFM import TSFM


def calculate_regression_metrics(y_true, y_pred):
    """
    Scikit-learn tabanlı regresyon metrikleri.
    y_true ve y_pred: log-return dizileri.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE: log-return'ler sıfıra çok yakın olduğu için
    # sıfıra bölme riski var. Bunun yerine sadece sıfır olmayan
    # gerçek değerler üzerinden hesaplıyoruz.
    nonzero_mask = np.abs(y_true) > 1e-8
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = float("inf")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def calculate_financial_metrics(y_true_returns, y_pred_returns):
    """
    Finansal metrikler.
    y_true_returns ve y_pred_returns: log-return dizileri (günlük).

    Log-return'lerde yön tahmini:
      - y_true > 0  => fiyat yükseldi
      - y_true < 0  => fiyat düştü
    Strateji: tahmin edilen yöne göre long/short pozisyon aç.
    """
    # Yön tahmini (Direction prediction)
    actual_dir = np.sign(y_true_returns)
    pred_dir = np.sign(y_pred_returns)
    hit_ratio = np.mean(actual_dir == pred_dir)

    # Log-return'leri basit return'e çevir: r = exp(log_r) - 1
    actual_simple_returns = np.exp(y_true_returns) - 1
    pred_simple_dir = np.sign(y_pred_returns)  # Yön bilgisi log-return'den

    # Strateji: Tahmin yönüne göre long (+1) veya short (-1)
    strat_returns = pred_simple_dir * actual_simple_returns

    # Kümülatif getiri
    cum_strat = np.cumprod(1 + strat_returns)
    cum_bh = np.cumprod(1 + actual_simple_returns)

    total_strat_ret = cum_strat[-1] - 1
    total_bh_ret = cum_bh[-1] - 1

    # Sharpe Ratio (252 işlem günü üzerinden yıllıklandırılmış)
    mean_ret = np.mean(strat_returns)
    std_ret = np.std(strat_returns)
    sharpe = np.sqrt(252) * (mean_ret / std_ret) if std_ret > 0 else 0.0

    # Max Drawdown
    drawdowns = cum_strat / np.maximum.accumulate(cum_strat) - 1
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
    emb_path = "/Users/tugan_basaran/Desktop/Lessons/CS.401/Kodlar/Moirai/uni2ts/Embeddings/moirai_1_embeddings.pt"
    data_path = "/Users/tugan_basaran/Desktop/Lessons/CS.401/Kodlar/Moirai/uni2ts/data/first_ten_columns.csv"

    try:
        raw_emb_dict = torch.load(emb_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading {emb_path}: {e}")
        return

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

    
    optimizer = torch.optim.Adam
    loss_fn = nn.MSELoss()
    
    print("Starting Training...")
    EPOCHS= 100
    trainer.train(
        loss_func=loss_fn, optimizer=optimizer, epochs=EPOCHS, learning_rate= 1e-4
    )

    # ─── Evaluation ───
    print("Evaluating on Test Set...")
    trainer.model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            targets = batch.y

            preds = trainer.model(
                e_i=batch.e_i,
                node_feature=batch.x,
                edge_index=batch.edge_index,
                edge_weight=batch.edge_attr,
                target_idx=batch.target_idx,
                batch_ptr=batch.ptr,
            )

            y_true_list.extend(targets.cpu().numpy())
            y_pred_list.extend(preds.squeeze().cpu().numpy())

    y_true = np.array(y_true_list).flatten()
    y_pred = np.array(y_pred_list).flatten()

    # ─── Regression Metrics (Log-Return uzayında) ───
    print("\n--- Regression Metrics on Test Data (Log-Return Space) ---")
    reg_metrics = calculate_regression_metrics(y_true, y_pred)
    for k, v in reg_metrics.items():
        print(f"{k}: {v:,.6f}")

    # ─── Financial Metrics ───
    print("\n--- Financial Metrics on Test Data ---")
    fin_metrics = calculate_financial_metrics(y_true, y_pred)
    for k, v in fin_metrics.items():
        if "Ratio" in k or "Drawdown" in k:
            print(f"{k}: {v:,.4f}")
        else:
            print(f"{k}: {v:,.2%}")
            
    print(f"İlk 10 tahmin: {y_pred[:10]}")
    print(f"İlk 10 gerçek:  {y_true[:10]}")
    print(f"Tahmin std:     {y_pred.std():.6f}")
    print(f"Gerçek std:     {y_true.std():.6f}")

    print(f"\n✅ Test completed! ({len(y_true)} samples evaluated)")


if __name__ == "__main__":
    main()
