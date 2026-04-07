import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    TextDataset,
    Trainer,
    TrainingArguments,
)

from config import get_model_and_tokenizer


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

    dataset = TextDataset(tokenizer=tokenizer, file_path=args.data, block_size=128)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
