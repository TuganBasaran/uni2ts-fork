import os
import sys
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.TSFMTrainer import TSFMTrainer

class TestDataPreparation:
    def run_test(self):
        print("--- Testing Data Preparation ---")
        print("1. Creating mock dataset...")
        
        # Simulate index as strings (like CSV reader returns typically unless parse_dates=True)
        # Create 20 days so a 5-day rolling correlation won't fail
        dates = pd.date_range("2010-01-01", periods=20, freq='D').strftime("%Y-%m-%d").tolist()
        
        # Original df
        df = pd.DataFrame(index=dates)
        # Use random data so rolling correlation doesn't output NaN due to 0 variance
        df["CO1 Comdty"] = np.random.rand(20) * 10.0 + 1.0
        df["CO2 Comdty"] = np.random.rand(20) * 10.0 + 1.0
        df.index.name = "date"
        
        mock_csv_path = "mock_prices.csv"
        df.to_csv(mock_csv_path)
        
        # Mock embeddings for all but the last 2 days
        embedding_dict = {}
        for i in range(18): 
            date = dates[i]
            embedding_dict[date] = {
                "CO1 Comdty": np.random.rand(16, 384),
                "CO2 Comdty": np.random.rand(16, 384)
            }
            
        print("2. Initializing TSFMTrainer...")
        trainer = TSFMTrainer(device="cpu")
        
        print("3. Calling prepare_data()...")
        train_loader, test_loader = trainer.prepare_data(
            embedding_dict=embedding_dict,
            data_path=mock_csv_path,
            context_length=256,
            patch_len=16, # seq_len = 16
            target_col="CO1 Comdty",
            index_col="date",
            corr_period=5, # Short correlation period for the test
            threshold=0.1,
            train_test_split=0.6,
            batch_size=2
        )
        
        print("4. Verification!")
        
        # Let's ensure the very first 'y_train' maps to originally dates[1] value of CO1 Comdty
        print(f"Originally expected target for index 0 (t+1): {df['CO1 Comdty'].iloc[1]}")
        
        for batch_idx, batch in enumerate(train_loader):
            print(f"Batch {batch_idx + 1} - Nodes: {batch.num_nodes}, X shape: {batch.x.shape}, y shape: {batch.y.shape}")
            if batch_idx == 0:
                print(f"First target value generated: {batch.y[0].item()}")
            break
            
        # Optional: try running one training step if torch_geometric installed properly locally
        print("Calling dummy training step on model...")
        import torch.optim as optim
        import torch.nn as nn
        optimizer = optim.Adam(trainer.model.parameters(), lr=0.001)
        # Sadece 1 epoch test edelim hatasiz donecek mi diye:
        trainer.fit(train_loader, test_loader, optimizer, nn.MSELoss(), epochs=1)
            
        # Clean up
        if os.path.exists(mock_csv_path):
            os.remove(mock_csv_path)

if __name__ == "__main__":
    tester = TestDataPreparation()
    tester.run_test()
