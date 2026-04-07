# SovoGPT: Odia Language Large Language Model

SovoGPT is an experimental project to build and fine-tune Large Language Models (LLMs) specifically for the **Odia language** using consumer-grade hardware. It explores the entire pipeline from tokenizer training to instruction fine-tuning and agentic behaviors.

## 🚀 Features
- **Custom Tokenizer:** Trained specifically on Odia text for better token efficiency.
- **Efficient Training:** Optimized scripts for training on consumer GPUs (Mac M-series/NVIDIA RTX).
- **Multi-Stage Pipeline:** Includes pre-training, fine-tuning, and RLHF-style alignment (Safe/Instruct modes).
- **Agentic Capabilities:** Experimental "Router" and "Agent" models designed to handle complex queries.

## 📂 Project Structure
```
Directory structure:
└── sovopr-sovogpt/
    ├── README.md
    ├── requirements.txt
    ├── data/
    │   └── router_data.json
    ├── scripts/
    │   ├── 2_train_balanced.py
    │   ├── 2_train_pro.py
    │   ├── 2_train_router.py
    │   ├── fine_tune.py
    │   ├── fine_tune_final.py
    │   ├── mix_data.py
    │   ├── prepare_data.py
    │   ├── train.py
    │   ├── train_agent.py
    │   └── train_tokenizer.py
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
   ```

## 🏃‍♂️ Usage

### 1. Train the Tokenizer
```bash
python scripts/train_tokenizer.py
```

### 2. Pre-train the Base Model
```bash
python scripts/train.py
```

### 3. Chat with the Model
```bash
python src/chat.py
```

## 📊 Model Versions
- **SovoGPT-Base:** The foundation model trained on Odia Wikipedia.
- **SovoGPT-Instruct:** Fine-tuned for following instructions.
- **SovoGPT-Safe:** Aligned version to refuse harmful queries.

## 🤝 Contributing
Contributions are welcome! Please open an issue if you encounter bugs or have datasets to share.
