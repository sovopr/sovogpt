from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from config import get_model_and_tokenizer
import os

print("Loading polite model (sovogpt_instruct)...")
model, tokenizer = get_model_and_tokenizer()

# Load the polite version
if os.path.exists("./sovogpt_instruct"):
    model.from_pretrained("./sovogpt_instruct")
else:
    print("Warning: sovogpt_instruct not found, using final")
    model.from_pretrained("./sovogpt_final")

print("Loading SAFE chat data...")
if not os.path.exists("clean_chat_data.txt"):
    print("Error: clean_chat_data.txt not found!")
    exit()

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="clean_chat_data.txt", 
    block_size=128
)

training_args = TrainingArguments(
    output_dir="./sovogpt_v5_safe", 
    overwrite_output_dir=True,
    num_train_epochs=5,  # 5 epochs to really learn the new style
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    save_steps=500,
    save_total_limit=2,
    use_mps_device=True,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=dataset,
)

print("Training on Safe Dialogue...")
trainer.train()

trainer.save_model("./sovogpt_v5_safe")
tokenizer.save_pretrained("./sovogpt_v5_safe")
print("Done! Chatbot ready at ./sovogpt_v5_safe")