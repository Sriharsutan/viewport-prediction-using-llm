import os
import pandas as pd
import torch

def load_all_viewport_data(root_dir, window_size=7, predict_horizon=1):
    sequences, targets = [], []

    # Traverse 3 levels deep to find CSV files
    for folder1 in os.listdir(root_dir):
        path1 = os.path.join(root_dir, folder1)
        if not os.path.isdir(path1): continue

        for folder2 in os.listdir(path1):
            path2 = os.path.join(path1, folder2)
            if not os.path.isdir(path2): continue

            for folder3 in os.listdir(path2):
                path3 = os.path.join(path2, folder3)
                if not os.path.isdir(path3): continue

                for file in os.listdir(path3):
                    if file.endswith(".csv"):
                        file_path = os.path.join(path3, file)
                        try:
                            df = pd.read_csv(file_path, header=None)

                            # Ensure there are at least 4 columns
                            if df.shape[1] < 4:
                                print(f"Skipping {file_path}: not enough columns")
                                continue

                            df.columns = ['timestamp', 'roll', 'pitch', 'yaw']
                            df = df[['roll', 'pitch', 'yaw']].dropna()

                            if len(df) < window_size + predict_horizon:
                                print(f"Skipping {file_path}: too few rows")
                                continue

                            for i in range(len(df) - window_size - predict_horizon + 1):
                                seq = df.iloc[i:i+window_size].values
                                tgt = df.iloc[i+window_size+predict_horizon-1].values
                                sequences.append(seq)
                                targets.append(tgt)

                        except Exception as e:
                            print(f"Skipping {file_path} due to error: {e}")

    if not sequences:
        print("\nNo sequences found. Check if your folder/data is accessible.")
    else:
        print(f"\nTotal sequences collected: {len(sequences)}")

    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# # Call the function
# X, y = load_all_viewport_data("data/")
# print("\nFinal tensor shapes:")
# print("X:", X.shape)
# print("y:", y.shape)
