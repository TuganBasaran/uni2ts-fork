import os
import sys

import numpy as np
import pandas as pd  # type: ignore
import torch
import torch.nn as nn
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore

# Add the 'grad' directory to the python path so it can find TSFM.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.preprocessing import StandardScaler
from TSFM import TSFM  # type: ignore


class TSFMTrainer:
    def __init__(self, device="mps"):
        self.device = device
        self.model = None

    def prepare_data(
        self,
        data_path,
        embedding_dict,
        patch_len=None,
        target_column="CO1 Comdty",
        index_col="date",
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        batch_size=32,
    ):
        self.data_path = data_path
        self.df = pd.read_csv(data_path, index_col=index_col)

        self.target_column = target_column
        self.embedding_dict = embedding_dict
        self.dict_keys_list = list(self.embedding_dict.keys())
        self.embedding_shape = self.embedding_dict[self.dict_keys_list[0]]

        if patch_len is None:
            self.patch_len = 32
        else:
            self.patch_len = self.embedding_shape[
                0
            ]  # (patch_len, sequence_len) -> (patch_len)

        self.model = TSFM(
            data_path=self.data_path,
            context_length=512,
            patch_len=self.patch_len,
            index_col="date",
            corr_period=30,
            threshold=0.3,
            device=self.device,
            forecast_horizon=1,
        )

        self.corr_dict = self.model.graph_dict

        self.corr_dates = np.array(list(self.corr_dict.keys()))
        self.dict_dates = np.array(list(self.embedding_dict.keys()))
        self.shifted_df = self.df.shift(-1).dropna().copy()
        self.returns_df = ((self.shifted_df - self.df) / self.df).dropna().copy()
        self.shifted_dates = np.array(list(self.shifted_df.index))

        self.valid_dates = np.intersect1d(self.corr_dates, self.dict_dates)
        self.valid_dates = np.intersect1d(self.valid_dates, self.shifted_dates)

        self.valid_df = self.df.loc[self.valid_dates]

        data_list = []

        print(f"Preparing dataloaders with target column {self.target_column}")

        # Calculate the node index of target column
        col_order = self.df.columns.tolist()
        target_node_idx = col_order.index(self.target_column)

        total_length = len(self.valid_dates)
        train_len = int(total_length * train_size)

        # Fit scaler ONLY on the training portion to prevent data leakage
        train_dates = self.valid_dates[:train_len]
        # Target is now the percentage RETURN, not the absolute price!
        train_y_returns = self.returns_df.loc[
            train_dates, self.target_column
        ].values.reshape(-1, 1)
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(train_y_returns)

        for date in self.valid_dates:
            e_i = self.embedding_dict[date][self.target_column]
            # Flatten e_i to 1D so PyG batches it correctly as [batch_size, flat_dim]
            e_i = torch.tensor(e_i, dtype=torch.float32).flatten().unsqueeze(0)

            node_feature = self.corr_dict[date]["node_features"]
            edge_index = self.corr_dict[date]["edge_index"]
            edge_weight = self.corr_dict[date]["edge_weight"]

            # Scale Returns!
            y_return_raw = self.returns_df.loc[date, self.target_column]
            y_scaled = self.target_scaler.transform([[y_return_raw]])[0][0]
            y = torch.tensor(y_scaled, dtype=torch.float32)

            # Keep track of the current day price (t) to reconstruct predictions at evaluation
            price_t = self.df.loc[date, self.target_column]
            price_t_tensor = torch.tensor(price_t, dtype=torch.float32)

            data = Data(
                x=node_feature,
                edge_index=edge_index,
                edge_attr=edge_weight,
                e_i=e_i,
                y=y,
                price_t=price_t_tensor,
                target_idx=torch.tensor([target_node_idx], dtype=torch.long),
            )

            data_list.append(data)

        total_length = len(data_list)
        # 1. Uzunlukların (index sınırlarının) hesaplanması
        train_len = int(total_length * train_size)
        test_len = int(total_length * test_size)
        val_len = (
            total_length - train_len - test_len
        )  # DÜZELTME: train_size değil, train_len

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        # 2. Hata Kontrolleri ve DataLoaders
        if train_len <= 0 or test_len <= 0:
            raise Exception(
                "The train size or test size is less or equal to 0. Fix your train and test percentages!"
            )

        if val_len < 0:
            raise Exception(
                "The validation size is less than 0. Fix your train and test percentages!"
            )

        # 3. Listelerin Kesilmesi (Slicing) - Her zaman geçmişten geleceğe doğru
        # Train (ilk kısım)
        self.train_dataloader = DataLoader(
            data_list[:train_len],
            batch_size=batch_size,
            shuffle=False,
        )

        # Eğer validation 0 ise, kalan tüm son veriler doğrudan Test'e gider
        if val_len == 0:
            self.test_dataloader = DataLoader(
                data_list[train_len:],
                batch_size=batch_size,
                shuffle=False,
            )

        # Eğer validation varsa, aradaki kısmı Val'e, en son güncel kısmı Test'e atarız
        else:
            val_end_index = train_len + val_len
            self.val_dataloader = DataLoader(
                data_list[train_len:val_end_index],
                batch_size=batch_size,
                shuffle=False,
            )
            self.test_dataloader = DataLoader(
                data_list[val_end_index:],
                batch_size=batch_size,
                shuffle=False,
            )

        # En nihayetinde fonksiyon bu 3 loader'ı döndürmeli
        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def train(self, loss_func, optimizer, epochs, learning_rate=1e-4):
        print(f"Starting Training on {self.device} for {epochs} epochs...")

        # Phase 3: Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch in self.train_dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                # TSFM Forward Pass (tüm özellikleri yolluyoruz)
                preds = self.model(
                    e_i=batch.e_i,
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_weight=batch.edge_attr,
                    target_idx=batch.target_idx,
                    batch_ptr=batch.ptr,
                )

                # Loss (Tahmin ile Hedefi Kıyaslama)
                loss = loss_func(preds, batch.y.view(-1, 1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch.num_graphs

            train_loss /= len(self.train_dataloader.dataset)

            # Validation Step
            val_loss = 0.0
            if self.val_dataloader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch in self.val_dataloader:
                        batch = batch.to(self.device)
                        preds = self.model(
                            e_i=batch.e_i,
                            x=batch.x,
                            edge_index=batch.edge_index,
                            edge_weight=batch.edge_attr,
                            target_idx=batch.target_idx,
                            batch_ptr=batch.ptr,
                        )
                        loss = loss_func(preds, batch.y.view(-1, 1))
                        val_loss += loss.item() * batch.num_graphs
                val_loss /= len(self.val_dataloader.dataset)
                print(
                    f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}"
                )
                scheduler.step(val_loss)
            else:
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}")
                scheduler.step(train_loss)

        print("Training Completed!")
