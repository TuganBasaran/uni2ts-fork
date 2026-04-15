import networkx as nx  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch


class SpatioGraph:
    def __init__(self, data_path, index_col="date", corr_period=30, device="cpu"):
        self.data_path = data_path
        self.index_col = index_col
        self.corr_period = corr_period
        self.df = pd.read_csv(data_path, index_col="date")

        self.dtype = torch.float32 if device == "mps" else torch.long
        self.log_returns = np.log(self.df / self.df.shift(1)).dropna()
        self.static_corr = self.log_returns.corr()
        self.rolling_corr = self.log_returns.rolling(window=corr_period).corr()

    def get_temporal_graphs(self, threshold=0.3):
        self.graphs = []
        tickers = self.df.columns.tolist()

        for date in self.log_returns.index[self.corr_period - 1 :]:
            corr = self.rolling_corr.loc[date]

            if corr.isnull().any().any():
                raise Exception(f"Null data spotted: {date}")

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

    def get_pyg_edges(self, threshold=0.3):
        """
        Her tarih için edge_index ve edge_weight döner.
        Doğrudan GraphModule'e verilebilir format.
        """
        self.graph_dict = {}

        for date in self.log_returns.index[self.corr_period - 1 :]:
            corr = self.rolling_corr.loc[date]

            if corr.isnull().any().any():
                raise Exception(f"Null Data Spotted: {date}")

            adj = corr.values.copy()
            adj = np.where(np.abs(adj) >= threshold, adj, 0)
            # Ticker'ların kendileriyle olan correlation'ları sıfırla, modelin kafası karışmasın
            np.fill_diagonal(adj, 0)
            src, dst = np.nonzero(adj)
            edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
            edge_weight = torch.tensor(adj[src, dst], dtype=torch.float32)

            self.graph_dict[date] = {}
            self.graph_dict[date]["node_features"] = torch.tensor(
                adj, dtype=torch.float32
            )
            self.graph_dict[date]["edge_index"] = torch.tensor(
                edge_index, dtype=torch.long
            )
            self.graph_dict[date]["edge_weight"] = torch.tensor(
                edge_weight, dtype=torch.float32
            )

        return self.graph_dict
