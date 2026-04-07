import sys
from run_system import load_model, generate_reply, to_roman_odia

inputs = [
    "namaskar, kemiti achha?",
    "tume kie?",
    "tumara kama kana?",
    "aaji rati re kana khaiba darkar? suggest kara",
    "bhubaneswar ra weather kemiti achi?",
    "mate gote interesting odia gapa kuha",
    "tume odia english duita missi ki katha heipariba?",
    "movie dekhiba pain kichi suggestions deba?",
    "tume kounthi ru asichha?",
    "achha, ai artificial intelligence kana?",
    "odia sahitya bishayare jama kichi kuha",
    "tumara favorite khadya kana?",
    "mate kichi bhal advice dia",
    "mu tike udas achi aaji, kana karibi?",
    "dhanyabad, tume bahut katha kahila. bye!"
]

def main():
    print("Loading model for evaluation...")
    model, tokenizer, path = load_model()
    device = "mps"
    model.to(device)
    model.eval()
    print("Model loaded. Starting 15-turn conversation...")
    
    history = []
    
    for i, user_raw in enumerate(inputs):
        print(f"\nTurn {i+1} - You: {user_raw}")
        user_text = to_roman_odia(user_raw)
        reply = generate_reply(model, tokenizer, device, history, user_text)
        print(f"Sovogpt: {reply}")
        history.append((user_text, reply))

if __name__ == '__main__':
    main()
