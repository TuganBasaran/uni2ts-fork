import os
import sys

import numpy as np
import pandas as pd # type: ignore
import torch
import torch.nn as nn
from torch_geometric.data import Data # type: ignore
from torch_geometric.loader import DataLoader # type: ignore

# Add the 'grad' directory to the python path so it can find TSFM.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from TSFM import TSFM # type: ignore


class TSFMTrainer:
    def __init__(self, device="mps"):
        self.device = device
        self.model = None

    def prepare_data(
        self,
        embedding_dict: dict,
        data_path: str,
        context_length: int,
        patch_len: int,
        target_col="CO1 Comdty",
        index_col="date",
        corr_period=30,
        threshold=0.3,
        forecast_horizon=1,
        train_test_split=0.8,
        batch_size=32,
    ):

        # TSFM modelini parametrelerle initialize ediyoruz
        self.model = TSFM(
            data_path=data_path,
            context_length=context_length,
            patch_len=patch_len,
            index_col=index_col,
            corr_period=corr_period,
            threshold=threshold,
            device=self.device,
            forecast_horizon=forecast_horizon,
        )
        self.model.to(self.device)

        self.embedding_dict = embedding_dict
        self.embedding_df = pd.DataFrame.from_dict(embedding_dict, orient="index")
        self.data_path = data_path
        self.target_col = target_col
        self.index_col = index_col
        # Read true values
        full_df = pd.read_csv(self.data_path, index_col=self.index_col)
        self.batch_size = batch_size
        self.train_test_split = train_test_split

        # Sadece embedding_dict içindeki "date" indekslerine karşılık gelen ancak gerçek veride var olanları çekiyoruz.
        valid_index = self.embedding_df.index.intersection(full_df.index)
        self.embedding_df = self.embedding_df.loc[valid_index]
        self.true_df = full_df.loc[valid_index, target_col]

        # Forecast horizon = 1: Bugünün embeddingleri ile yarının hedefini eşleştirmek için bir geri al:
        self.true_df = self.true_df.shift(-1).dropna()

        valid_dates = self.true_df.index

        # Sütun sırasını belirleyelim (node sırası). İlk tarihteki embedding dictionary key'lerini referans al.
        first_date = valid_dates[0]
        col_order = list(self.embedding_dict[first_date].keys())
        target_node_idx = col_order.index(target_col)

        data_list = [] #type: ignore

        for date in valid_dates:
            if date not in self.model.graph_dict:
                continue  # Korelasyon olmayan ilk günleri atla

            embeddings_for_date = []

            # Düğümleri her zaman aynı sırada tensor'a dönüştür.
            for col in col_order:
                emb = self.embedding_dict[date][col]
                embeddings_for_date.append(emb)

            # Beklenen Node Shape: (num_nodes, seq_len, 384) -> Flatten(num_nodes, seq_len * 384) PyG için
            e_stacked = np.stack(embeddings_for_date, axis=0)
            e_stacked_flat = e_stacked.reshape(e_stacked.shape[0], -1)

            # TSFM modelindeki boyuta uyum kontrolü
            if len(data_list) == 0:
                assert e_stacked.shape[1] == self.model.seq_len, (
                    f"Embedding seq_len {e_stacked.shape[1]} doesn't match model seq_len {self.model.seq_len}"
                )
                assert e_stacked.shape[2] == 384, (
                    f"Embedding dim {e_stacked.shape[2]} must be 384"
                )

            x_tensor = torch.tensor(e_stacked_flat, dtype=torch.float32)
            y_tensor = torch.tensor([self.true_df.loc[date]], dtype=torch.float32)

            edge_index = self.model.graph_dict[date]["edge_index"]
            edge_weight = self.model.graph_dict[date]["edge_weight"]

            # Create PyTorch Geometric Data object
            data = Data(
                x=x_tensor,
                edge_index=edge_index,
                edge_attr=edge_weight,
                y=y_tensor,
                target_idx=torch.tensor([target_node_idx], dtype=torch.long),
            )
            data.num_nodes_custom = len(col_order)
            data_list.append(data)

        total_size = len(data_list)
        if total_size == 0:
            raise ValueError(
                "No valid graphs could be parsed! (Check your correlation period and data size)"
            )

        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)

        # Train, Val, Test Bölülmesi
        train_data = data_list[:train_size]
        val_data = data_list[train_size:train_size + val_size]
        test_data = data_list[train_size + val_size:]

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        print("✅ Data preparation successful.")
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")

        return train_loader, val_loader, test_loader

    def fit(self, train_loader, val_loader, optimizer, loss_fn, epochs=20):
        print(f"Starting Training on {self.device} for {epochs} epochs...")
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                # Forward Pass
                output = self.model(batch.x, batch.edge_index, batch.edge_attr)

                # Extract predictions belonging to target_col
                # batch.ptr stores the start index of every graph in the batch
                # batch.target_idx stores the node index of the target inside each local graph
                target_indices = batch.ptr[:-1] + batch.target_idx.flatten()
                target_preds = output[target_indices]

                # Calculate loss
                loss = loss_fn(target_preds, batch.y.view(-1, 1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch.num_graphs

            train_loss /= len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)

                    output = self.model(batch.x, batch.edge_index, batch.edge_attr)
                    target_indices = batch.ptr[:-1] + batch.target_idx.flatten()
                    target_preds = output[target_indices]

                    loss = loss_fn(target_preds, batch.y.view(-1, 1))
                    val_loss += loss.item() * batch.num_graphs

            val_loss /= len(val_loader.dataset)

            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}"
            )
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

        print("Training Completed!")
        return history
