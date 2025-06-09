import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from cnn_model import ViewportPredictor

class ViewportDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        with open(jsonl_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        enc = self.tokenizer(sample["prompt"], truncation=True, max_length=self.max_length, return_tensors="pt")
        dec = self.tokenizer(sample["completion"], truncation=True, max_length=self.max_length, return_tensors="pt")
        enc_ids = enc["input_ids"].squeeze(0)
        dec_ids = dec["input_ids"].squeeze(0)
        return {"input_ids": enc_ids, "labels": dec_ids}

if __name__ == "__main__":
    model = ViewportPredictor()
    tokenizer = model.tokenizer

    dataset = ViewportDataset("viewport_data.json", tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=10,
        save_total_limit=2,
        save_steps=50,
        evaluation_strategy="no",
        fp16=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
