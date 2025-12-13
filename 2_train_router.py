from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
import json
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

print("--- Step 2: Training the Router Brain ---")

# 1. Load Data
try:
    with open("router_data.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: router_data.json not found. Run Step 1 first.")
    exit()

dataset = Dataset.from_dict(data)

# 2. Load Base Model (Sentence Transformer)
print("Loading Base Model...")
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")

# 3. Train
print("Training Neural Router (This is fast)...")
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./sovogpt_router",
        num_epochs=5,
        batch_size=4
    )
)

trainer.train()
model.save_pretrained("./sovogpt_router")
print("\n[SUCCESS] Router trained and saved to ./sovogpt_router. Proceed to Step 3.")