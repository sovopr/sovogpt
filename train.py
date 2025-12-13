# train.py
import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, TextDataset
from config import get_model_and_tokenizer

# 1. Setup Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load Model & Tokenizer
model, tokenizer = get_model_and_tokenizer()
model.to(device)

# 3. Prepare Dataset (Streaming approach for memory safety)
print("Loading dataset into memory blocks...")
# Block size = how long of a sentence the model can see at once
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="train_data.txt",
    block_size=128 
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# 4. Training Arguments (TUNED FOR MAC M2 16GB)
training_args = TrainingArguments(
    output_dir="./sovogpt_checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=3,             # How many times to read the whole file
    per_device_train_batch_size=8,  # Keep low to save RAM
    gradient_accumulation_steps=4,  # Effective batch size = 32
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    use_mps_device=True,            # Force Mac GPU usage
    learning_rate=3e-4,
    gradient_checkpointing=True,    # SAVES MASSIVE RAM
    logging_steps=50,
)

# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 6. Train
print("Starting training... Go grab a coffee (or sleep).")
trainer.train()

# 7. Save Final Model
trainer.save_model("./sovogpt_final")
tokenizer.save_pretrained("./sovogpt_final")
print("Training Complete! Model saved to ./sovogpt_final")