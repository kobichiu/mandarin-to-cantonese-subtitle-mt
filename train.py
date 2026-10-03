import argparse
import csv
import json

import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

MODEL_NAME = "google/mt5-small"
EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MAX_LENGTH = 128
SEED = 42
EVAL_STEPS = 500
NUM_BEAMS = 4


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                print(f"Skipping invalid line {line_number}")
    return records


class SubtitlesDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=MAX_LENGTH):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        input_enc = self.tokenizer(
            record["input"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        target_enc = self.tokenizer(
            record["target"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        labels = [
            token if token != self.tokenizer.pad_token_id else -100
            for token in target_enc["input_ids"]
        ]
        return {
            "input_ids": torch.tensor(input_enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(input_enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def generate_predictions(model, tokenizer, records, device):
    model.to(device)
    predictions = []
    for record in records:
        encoded = tokenizer(
            record["input"],
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model.generate(
            **encoded,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
        )
        predictions.append(tokenizer.decode(output[0], skip_special_tokens=True))
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="Training set written by split_corpus.py")
    parser.add_argument("dev", help="Development set written by split_corpus.py")
    parser.add_argument("--test", help="Test set; predictions are written if given")
    args = parser.parse_args()

    set_seed(SEED)
    train_records = load_jsonl(args.train)
    dev_records = load_jsonl(args.dev)
    print(f"Loaded train {len(train_records)}, dev {len(dev_records)}")

    tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./mt5_small_cantonese",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=EVAL_STEPS,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        generation_num_beams=NUM_BEAMS,
        report_to="none",
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=SubtitlesDataset(train_records, tokenizer),
        eval_dataset=SubtitlesDataset(dev_records, tokenizer),
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()

    if args.test:
        test_records = load_jsonl(args.test)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictions = generate_predictions(trainer.model, tokenizer, test_records, device)
        with open("predictions.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["input", "target", "predicted"])
            writer.writeheader()
            for record, prediction in zip(test_records, predictions):
                writer.writerow(
                    {
                        "input": record["input"],
                        "target": record["target"],
                        "predicted": prediction,
                    }
                )
        print("Saved predictions.csv")

    trainer.model.save_pretrained("./mt5_small_finetuned")
    tokenizer.save_pretrained("./mt5_small_finetuned")
    print("Saved model to ./mt5_small_finetuned")


if __name__ == "__main__":
    main()
