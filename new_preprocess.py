import os
import json
import pandas as pd
import argparse

def generate_samples(csv_path, past=5, future=5, step=1):
    df = pd.read_csv(csv_path, header=None, names=["timestamp", "roll", "pitch", "yaw"])
    df = df.dropna()
    
    data = df[["roll", "pitch", "yaw"]].values
    samples = []

    for i in range(0, len(data) - (past + future), step):
        past_data = data[i:i+past]
        future_data = data[i+past:i+past+future]

        prompt = f"The past {past} viewports were:\n" + "\n".join(
            [f"({r:.3f},{p:.3f},{y:.3f})" for r, p, y in past_data]
        ) + f"\nWhat are the next {future} viewports?"

        completion = "\n" + "\n".join(
            [f"({r:.3f},{p:.3f},{y:.3f})" for r, p, y in future_data]
        )

        samples.append({"prompt": prompt, "completion": completion})

    return samples

def process_all(datasets_dir, out_file, past, future):
    all_samples = []
    for root, _, files in os.walk(datasets_dir):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(root, file)
                all_samples.extend(generate_samples(path, past, future))
    with open(out_file, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")
    print(f"✅ Done: {len(all_samples)} samples written to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="datasets/Jin2022")
    parser.add_argument("--out", type=str, default="viewport_data.json")
    parser.add_argument("--past", type=int, default=5)
    parser.add_argument("--future", type=int, default=5)
    args = parser.parse_args()
    process_all(args.datasets, args.out, args.past, args.future)
