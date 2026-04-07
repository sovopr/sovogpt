import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import get_model_and_tokenizer


class ChatMLDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
        conversations = [c.strip() for c in data.split("<|im_start|>system\n") if c.strip()]
        self.examples = []
        for c in conversations:
            text = "<|im_start|>system\n" + c
            parts = text.split("<|im_start|>assistant\n")
            if len(parts) == 2:
                prompt = parts[0] + "<|im_start|>assistant\n"
                response = parts[1]
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
                
                input_ids = prompt_ids + resp_ids
                labels = [-100]*len(prompt_ids) + resp_ids
                
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                    labels = labels[:max_length]
                    
                self.examples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long)
                })
                
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]

class ChatDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sovogpt with a single optimizer strategy.")
    parser.add_argument("--data", default="agent_training_data.txt", help="Training data path")
    parser.add_argument("--output", default="./sovogpt_agent_model", help="Model output directory")
    parser.add_argument("--epochs", type=int, default=6, help="Epoch count")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument(
        "--base-model",
        default="./sovogpt_agent_model",
        help="Base model path for continuation fine-tuning. Falls back to config init if missing.",
    )
    parser.add_argument("--max-steps", type=int, default=-1, help="Override total training steps (-1 uses epochs)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Training file not found: {args.data}")

    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    use_mps = device_name == "mps"
    print(f"Using device: {device_name}")

    if os.path.exists(args.base_model):
        print(f"Loading base model from {args.base_model}")
        model = LlamaForCausalLM.from_pretrained(args.base_model)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.base_model)
    else:
        print("Base model not found, initializing model/tokenizer from config.py")
        model, tokenizer = get_model_and_tokenizer()
    model.to(device_name)

    dataset = ChatMLDataset(file_path=args.data, tokenizer=tokenizer, max_length=256)
    collator = ChatDataCollator(tokenizer=tokenizer)

    # Keep one clear optimizer path: AdamW (torch implementation).
    train_args = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        use_mps_device=use_mps,
        max_steps=args.max_steps,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=dataset,
    )

    print("Training Sovogpt agent...")
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
