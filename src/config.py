# config.py
import os
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

def get_model_and_tokenizer():
    # Resolve tokenizer file relative to project root
    tok_path = os.path.join(os.path.dirname(__file__), "..", "sovogpt_tokenizer.json")
    if not os.path.exists(tok_path):
        tok_path = "sovogpt_tokenizer.json"

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tok_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"]
    )
    tokenizer.pad_token = "<pad>"

    # TinyLlama Architecture
    config = LlamaConfig(
        vocab_size=16000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
    )
    
    model = LlamaForCausalLM(config)
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = get_model_and_tokenizer()
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")