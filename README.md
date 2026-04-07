# SovoGPT: Odia Language Large Language Model

SovoGPT is an experimental project to build and fine-tune Large Language Models (LLMs) specifically for the **Odia language** using consumer-grade hardware. It explores the entire pipeline from tokenizer training to instruction fine-tuning and agentic behaviors.

## 🚀 Features
- **Custom Tokenizer:** Trained specifically on Odia text for better token efficiency.
- **Efficient Training:** Optimized scripts for training on consumer GPUs (Mac M-series/NVIDIA RTX).
- **Multi-Stage Pipeline:** Includes pre-training, fine-tuning, and RLHF-style alignment (Safe/Instruct modes).
- **Agentic Capabilities:** Experimental "Router" and "Agent" models designed to handle complex queries.

## 📂 Project Structure
```
sovogpt/
├── .gitignore
├── README.md
├── agent_training_data.txt
├── balanced_training.txt
├── chat_boost_training_data.txt
├── clean_chat_data.txt
├── data/
│   └── router_data.json
├── final_mix.txt
├── massive_training_data.txt
├── odinglish_prompt_answer_long_220.csv
├── odinglish_prompt_answer_summary.csv
├── odinglish_prompt_answer_summary.md
├── real_chat_dataset.txt
├── requirements.txt
├── result.txt
├── scripts/
│   ├── prepare_odinglish_data.py
│   ├── run_system.py
│   ├── test_eval.py
│   ├── train_agent.py
│   └── train_tokenizer.py
├── sovogpt_tokenizer-merges.txt
├── sovogpt_tokenizer-vocab.json
├── sovogpt_tokenizer.json
├── src/
│   ├── chat.py
│   ├── chat_agent.py
│   ├── chat_internet.py
│   └── config.py
├── synthetic_chat.txt
├── system_error.log
└── train_data.txt
```
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
