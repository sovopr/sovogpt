# train_tokenizer.py
from tokenizers import ByteLevelBPETokenizer

# Initialize
tokenizer = ByteLevelBPETokenizer()

# Train
print("Training tokenizer...")
tokenizer.train(files=["train_data.txt"], vocab_size=16000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# FIX: Save as a single JSON file instead of separate vocab/merges
tokenizer.save("sovogpt_tokenizer.json")
print("Tokenizer saved as sovogpt_tokenizer.json!")