from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset

# 1. Define The Logic Patterns (Teaching the AI concepts, not hardcoding strings)
# The AI uses these examples to generalize to unseen sentences.
data = {
    "text": [
        # --- IDENTITY (Who are YOU?) ---
        "tu kie", "tame kie", "apana kie", "tuma na kana", "tor na kana", 
        "who are you", "what is your name", "identify yourself", "intro",
        
        # --- CHAT (General Conversation) ---
        "namaskar", "hi", "hello", "kemiti acha", "kemiti achanti", 
        "kana karucha", "what are you doing", "bhala achi", "good morning",
        "mote bhoka laguchi", "help me", "moro help karipariba", "gote story kuha",
        "tame kana kana kari pariba",
        
        # --- WEATHER (The Weather Tool) ---
        "weather", "weather in bhubaneswar", "puri re weather kimiti", 
        "aaji varsha heba ki", "temperature kete", "paga", "tapamatra",
        "is it raining", "forecast",
        
        # --- SEARCH (Facts about the World) ---
        "odisha ra cm kiye", "who is the pm", "india capital", "bhubaneswar kana",
        "population of india", "cricket score", "gold price", "news",
        "where is cuttack", "meaning of ai", "konark mandira", "jagannath temple"
    ],
    "label": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, # Identity
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # Chat
        2, 2, 2, 2, 2, 2, 2, 2, 2, # Weather
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 # Search
    ]
}

# 2. Train the Neural Network
print("Training The Cortex (Sentence Transformer)...")
dataset = Dataset.from_dict(data)
# This uses a pre-trained BERT model optimized for logic
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(output_dir="./sovogpt_cortex", num_epochs=10, batch_size=4)
)

trainer.train()
model.save_pretrained("./sovogpt_cortex")
print("Cortex Trained. It can now distinguish 'Who is CM' from 'Who are you'.")