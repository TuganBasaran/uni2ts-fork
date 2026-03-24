import networkx as nx  # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore



class SpatioGraph:
    def __init__(self, data_path, index_col="date", corr_period=30):
        self.data_path = data_path
        self.index_col = index_col
        self.corr_period = corr_period
        self.df = pd.read_csv(data_path, index_col="date")
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