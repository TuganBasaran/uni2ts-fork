import networkx as nx  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch


class SpatioGraph:
    def __init__(self, data_path, index_col="date", corr_period=30, device="cpu", target_col="CO1 Comdty"):
        self.data_path = data_path
        self.index_col = index_col
        self.corr_period = corr_period
        self.df = pd.read_csv(data_path, index_col="date")

        self.dtype = torch.float32 if device == "mps" else torch.long
        self.target_col = target_col

        # 1. Log-return hesapla
        self.log_returns = np.log(self.df / self.df.shift(1))
        # 2. inf'leri NaN'a çevir ve at
        self.log_returns = self.log_returns.replace([np.inf, -np.inf], np.nan).dropna()
        # 3. Tamamen sıfır satırları at (hafta sonu/tatil)
        non_zero_mask = (self.log_returns != 0).any(axis=1)
        self.log_returns = self.log_returns[non_zero_mask]
        # 4. Outlier'ları clip'le (±10%)
        self.log_returns = self.log_returns.clip(lower=-0.10, upper=0.10)

        self.static_corr = self.log_returns.corr()
        self.rolling_corr = self.log_returns.rolling(window=corr_period).corr()

    def get_temporal_graphs(self, threshold=0.3):
        self.graphs = []
        tickers = self.df.columns.tolist()

        for date in self.log_returns.index[self.corr_period - 1 :]:
            corr = self.rolling_corr.loc[date]

            if corr.isnull().any().any():
                # raise Exception(f"Null data spotted: {date}")
                continue

            G = nx.Graph()

            for i, ticker in enumerate(tickers):
                G.add_node(ticker, feature=self.log_returns.loc[date, ticker])

            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    weight = corr.iloc[i, j]
                    if abs(weight) >= threshold:
                        G.add_edge(tickers[i], tickers[j], weight=weight)

            self.graphs.append((date, G))

        return self.graphs

    def get_pyg_edges(self, threshold=0.3, lookback=5):
        self.graph_dict = {}

        for date in self.log_returns.index[self.corr_period - 1:]:
            corr = self.rolling_corr.loc[date]

            if corr.isnull().any().any():
                continue

            adj = corr.values.copy()
            adj = np.where(adj >= threshold, adj, 0)
            np.fill_diagonal(adj, 0)

            src, dst = np.nonzero(adj)
            edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
            edge_weight = torch.tensor(adj[src, dst], dtype=torch.float32)

            # Son N günlük log-return'leri al
            date_idx = self.log_returns.index.get_loc(date)
            if date_idx < lookback:
                continue

            # Her node için son 'lookback' günlük getirileri feature olarak ver
            node_features = torch.tensor(
                self.log_returns.iloc[date_idx - lookback + 1 : date_idx + 1].values.T,
                dtype=torch.float32
            )  # (num_nodes, lookback)
            
            # node_features = torch.tensor(self.log_returns.loc[date, self.target_col], dtype= torch.float32)

            self.graph_dict[date] = {
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_weight": edge_weight,
            }

        return self.graph_dict