from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from config import get_model_and_tokenizer
import os

# 1. Load the "Sane" Model (Layer 2 - Manners)
# We want to start with the model that knows how to be polite, 
# NOT the one that thinks everything is a number.
print("Loading base model (sovogpt_instruct)...")
model, tokenizer = get_model_and_tokenizer()

model_path = "./sovogpt_instruct" 
if os.path.exists(model_path):
    model.from_pretrained(model_path)
else:
    print("Warning: Could not find sovogpt_instruct. Falling back to sovogpt_final.")
    model.from_pretrained("./sovogpt_final")

# 2. Load the MIXED Data (Logic + Manners)
# This file must exist! (Run mix_data.py first if you haven't)
print("Loading mixed dataset (final_mix.txt)...")
if not os.path.exists("final_mix.txt"):
    print("ERROR: final_mix.txt not found! Did you run python mix_data.py?")
    exit()

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="final_mix.txt", 
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 3. Training Config
training_args = TrainingArguments(
    output_dir="./sovogpt_complete", # <--- The Final Output Folder
    overwrite_output_dir=True,
    num_train_epochs=3,              # 3 Epochs is perfect for mixed data
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    save_steps=500,
    save_total_limit=2,
    use_mps_device=True,             # Mac GPU
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 4. Run Training
print("Starting Final 'Mixer' Training...")
trainer.train()

# 5. Save the Ultimate Model
trainer.save_model("./sovogpt_complete")
tokenizer.save_pretrained("./sovogpt_complete")
print("Done! Your Complete Chatbot is ready at ./sovogpt_complete")