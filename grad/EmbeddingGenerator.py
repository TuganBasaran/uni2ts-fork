import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
from uni2ts.model.moirai2 import Moirai2Module  # type: ignore


class EmbeddingGenerator:
    def __init__(self, data_path, batch_size, context_length, patch_len):
        self.data_path = data_path
        self.batch_size = batch_size
        self.context_length = context_length
        self.patch_len = patch_len
        self.seq_len = context_length // patch_len
        self.device = "cpu"

        self.module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
        self.module.to(self.device)

        self.set_data()

        print(f"Sequence Length set at: {self.seq_len}")
        print(f"Device set: {self.device}")

    def set_data(self):
        self.df = pd.read_csv(self.data_path, index_col="date")

        self.columns = self.df.columns.values[0:10]
        print(f"Length of columns: {len(self.columns)}")

        self.data_dict = {}

        for column in self.columns:
            self.data_dict[column] = self.df[column].values

    def get_embeddings(self, technique="Sliding Window"):
        """
        This method employs rolling or sliding window technique
        to generate embeddings with specified hyperparameters:
            - Context Length
            - Patch Length
            - Batch Size

        Technique:
        0 -> Rolling Forecast / Expanding Window
        1 -> Sliding Window
        """

        dates = self.df.index
        self.embedding_dict = {}

        sample_id = torch.zeros((self.batch_size, self.seq_len), dtype=torch.long).to(
            self.device
        )
        time_id = (
            torch.arange(0, self.seq_len, dtype=torch.long)
            .unsqueeze(0)
            .expand(self.batch_size, -1)
        ).to(self.device)
        prediction_mask = torch.zeros(
            (self.batch_size, self.seq_len), dtype=torch.bool
        ).to(self.device)
        training_mode = False

        if technique == "Sliding Window":
            for col_idx, column in enumerate(self.columns):
                values = self.data_dict[column]
                variate_id = torch.full(
                    [self.batch_size, self.seq_len], col_idx, dtype=torch.long
                ).to(self.device)

                for i in range(len(values) - self.context_length + 1):
                    date = dates[i + self.context_length - 1]

                    target_np = values[i : i + self.context_length]
                    target_tensor = torch.tensor(target_np, dtype=torch.float32).to(
                        self.device
                    )
                    target_tensor = target_tensor.reshape(
                        self.batch_size, self.seq_len, self.patch_len
                    )

                    observed_mask = ~torch.isnan(target_tensor)

                    with torch.inference_mode():
                        embedding = self.module.forward(
                            target=target_tensor,
                            observed_mask=observed_mask,
                            sample_id=sample_id,
                            time_id=time_id,
                            variate_id=variate_id,
                            prediction_mask=prediction_mask,
                            training_mode=training_mode,
                        )

                        # Embedding Pooling Yapılmalı mı? Hocaya sor -> mean pooling? Son index'ini al???
                        # Şu anlık pooling yok full embedding kullanılarak
                        embedding = embedding.squeeze(0).cpu().numpy()

                        if date not in (self.embedding_dict):
                            self.embedding_dict[date] = {}

                        self.embedding_dict[date][column] = embedding

            return self.embedding_dict
