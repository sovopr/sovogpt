from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from config import get_model_and_tokenizer
import os

print("Loading Instruct Model...")
model, tokenizer = get_model_and_tokenizer()

# STARTING POINT: Your existing polite model
if os.path.exists("./sovogpt_instruct"):
    model.from_pretrained("./sovogpt_instruct")
else:
    model.from_pretrained("./sovogpt_final")

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="agent_training_data.txt", 
    block_size=128
)

training_args = TrainingArguments(
    output_dir="./sovogpt_agent_model", 
    overwrite_output_dir=True,
    num_train_epochs=3, 
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

print("Training Sovogpt Agent...")
trainer.train()
trainer.save_model("./sovogpt_agent_model")
tokenizer.save_pretrained("./sovogpt_agent_model")