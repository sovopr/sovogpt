import json
import random
from prepare_odinglish_data import to_roman_odia, SYSTEM_INSTRUCTION

def main():
    print("Loading jsonl...")
    pairs = []
    with open("data/odinglish_conversations.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            conv = json.loads(line)
            if len(conv) == 2 and conv[0]['role'] == 'user' and conv[1]['role'] == 'assistant':
                u = to_roman_odia(conv[0]['content'])
                a = to_roman_odia(conv[1]['content'])
                if len(u) > 1 and len(a) > 1:
                    pairs.append((u, a))
                    
    print(f"Loaded {len(pairs)} pairs.")
    # Shuffle so that multi-turn combinations are diverse
    random.shuffle(pairs)
    
    chunk_size = 5
    output_rows = []
    
    for i in range(0, len(pairs), chunk_size):
        chunk = pairs[i:i+chunk_size]
        if len(chunk) == 0: continue
        
        conv_text = f"<|im_start|>system\n{SYSTEM_INSTRUCTION}<|im_end|>\n"
        for user, assistant in chunk:
            conv_text += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n"
            
        output_rows.append(conv_text)
        
    print(f"Generated {len(output_rows)} multi-turn conversations.")
    with open("agent_training_data.txt", "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(row)
            
if __name__ == '__main__':
    main()
