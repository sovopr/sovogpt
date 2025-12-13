from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from config import get_model_and_tokenizer
import os

print("Loading Instruct Model...")
model, tokenizer = get_model_and_tokenizer()

if os.path.exists("./sovogpt_instruct"):
    model.from_pretrained("./sovogpt_instruct")
else:
    model.from_pretrained("./sovogpt_final")

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="balanced_training.txt", 
    block_size=128
)

training_args = TrainingArguments(
    output_dir="./sovogpt_balanced_agent", # New Model Name
    overwrite_output_dir=True,
    num_train_epochs=3, # 3 is enough for balanced data
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    save_steps=500,
    save_total_limit=2,
    use_mps_device=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=dataset,
)

print("Training Balanced Agent...")
trainer.train()
trainer.save_model("./sovogpt_balanced_agent")
tokenizer.save_pretrained("./sovogpt_balanced_agent")