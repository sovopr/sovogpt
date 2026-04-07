# config.py
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

def get_model_and_tokenizer():
    # Load our custom tokenizer (Points to the new single file)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="sovogpt_tokenizer.json", # <--- UPDATED THIS LINE
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>"
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