from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from config import get_model_and_tokenizer
import os

print("Loading Sovogpt Instruct...")
model, tokenizer = get_model_and_tokenizer()

# We start from 'instruct' because it's a good base
if os.path.exists("./sovogpt_instruct"):
    model.from_pretrained("./sovogpt_instruct")
else:
    model.from_pretrained("./sovogpt_final")

print("Loading MASSIVE Dataset...")
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="massive_training_data.txt", 
    block_size=128
)

training_args = TrainingArguments(
    output_dir="./sovogpt_groundbreaking", 
    overwrite_output_dir=True,
    num_train_epochs=10,             # Deep learning (takes time)
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,             # Lower, slower, better
    weight_decay=0.01,              # Prevents overfitting/looping
    lr_scheduler_type="cosine",     # Advanced scheduler
    save_steps=1000,
    save_total_limit=2,
    use_mps_device=True,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=dataset,
)

print("Starting Deep Training (This will take hours)...")
trainer.train()
trainer.save_model("./sovogpt_groundbreaking")
tokenizer.save_pretrained("./sovogpt_groundbreaking")
print("Done. You now have a Pro Model.")