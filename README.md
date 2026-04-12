# SovoGPT: Odia Language Large Language Model

SovoGPT is an experimental project to build and fine-tune Large Language Models (LLMs) specifically for the **Odia language** using consumer-grade hardware. It explores the entire pipeline from tokenizer training to instruction fine-tuning.

## 🚀 Features
- **Custom Tokenizer:** Trained specifically on Odia text for better token efficiency.
- **nanochat Engine Integration:** Shifted to a minimal, hackable GPT-2 pipeline for end-to-end training (pre-training → SFT → inference).
- **M2 Pro Optimized:** Native MPS (Metal Performance Shaders) support for high-speed training on Mac GPUs.
- **Odinglish Support:** Native Odia+English conversational capabilities with automated transliteration.

## 📂 Project Structure
```
sovogpt/
├── README.md
├── TECH_STACK.md
├── requirements.txt
├── nanochat_engine/      # Core GPT-2 training engine
├── data/                 # Training datasets (JSONL/TXT)
│   ├── odinglish_conversations.jsonl
│   ├── odinglish_pretrain.txt
│   └── legacy/           # Historical source data
├── runs/
│   └── odinglish.sh       # End-to-end training pipeline
├── scripts/
│   ├── convert_to_nanochat.py
│   ├── sovogpt_chat.py    # CLI chat interface
│   ├── sovogpt_eval.py    # Automated evaluation suite
│   └── ... (utility scripts)
└── src/
    ├── chat.py
    └── config.py
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sovopr/sovogpt.git
   cd sovogpt
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   cd nanochat_engine && uv sync && cd ..
   ```

## 🏃‍♂️ Usage

### 1. Data Conversion
Convert raw ChatML data to nanochat format:
```bash
python scripts/convert_to_nanochat.py
```

### 2. Full Pipeline (Training)
Run the automated pre-training and SFT pipeline:
```bash
bash runs/odinglish.sh
```

### 3. Chat with the Model
Start the interactive Odinglish chat:
```bash
python scripts/sovogpt_chat.py
```

## 📊 Model Versions
- **SovoGPT-Base:** The foundation model trained on Odia Wikipedia.
- **SovoGPT-Instruct:** Fine-tuned for following instructions.
- **SovoGPT-Safe:** Aligned version to refuse harmful queries.

## 🤝 Contributing
Contributions are welcome! Please open an issue if you encounter bugs or have datasets to share!
