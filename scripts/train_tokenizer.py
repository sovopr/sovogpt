# train_tokenizer.py
import os
import sys
from tokenizers import ByteLevelBPETokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Locate training files
candidates = [
    os.path.join(PROJECT_ROOT, "data", "conversational_odinglish.txt"),
    os.path.join(PROJECT_ROOT, "data", "odinglish_pretrain.txt"),
    os.path.join(PROJECT_ROOT, "data", "legacy", "agent_training_data.txt"),
    os.path.join(PROJECT_ROOT, "agent_training_data.txt"),
]
training_files = [f for f in candidates if os.path.exists(f)]

if not training_files:
    print("ERROR: No training corpus found in data/ or root directory.")
    sys.exit(1)

print(f"Training tokenizer on: {training_files[0]}")
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[training_files[0]],
    vocab_size=16000,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
        "<|im_start|>",
        "<|im_end|>",
    ],
)

output_path = os.path.join(PROJECT_ROOT, "sovogpt_tokenizer.json")
tokenizer.save(output_path)
print(f"Tokenizer saved to {output_path}!")