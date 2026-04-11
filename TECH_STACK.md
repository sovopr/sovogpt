# Sovogpt × nanochat: Technical Stack Overview

This document provides a deep dive into the architecture to understand the project structure and design decisions.

## 🧠 Core Architecture Shift
We have migrated from a **Hugging Face Llama** stack to a custom **nanochat GPT-2** stack.
- **Old Stack:** `transformers` library, `LlamaForCausalLM`, `Trainer` API, ChatML format.
- **New Stack:** `nanochat` engine (pure PyTorch), GPT-2 architecture, custom KV-cached `Engine`, JSONL conversation format.

## 🛠️ Components

### 1. Data Pipeline (`scripts/convert_to_nanochat.py`)
- **Input:** `agent_training_data.txt` (Hugging Face / ChatML format).
- **Transformation:** Extracts user/assistant pairs, performs Odinglish cleaning, and outputs:
  - `data/odinglish_conversations.jsonl`: for SFT.
  - `data/odinglish_pretrain.txt`: for tokenizer and base training.

### 2. Training Engine (`nanochat_engine/`)
- A clone of Karpathy's `nanochat`.
- **Customizations for Sovogpt:**
  - `runs/odinglish.sh`: A customized training script for Mac M2 Pro (MPS).
  - Uses `NANOCHAT_DTYPE=float32` because MPS does not support `bfloat16`.
  - Tuned for a `depth-6` model to allow rapid iteration on local hardware.

### 3. Inference Layer (`scripts/sovogpt_chat.py`)
- **Transliteration:** Uses `indic_transliteration` (sanscript) to convert Odia script to Roman script (Odinglish).
- **Tokenization:** Uses nanochat's `RustBPETokenizer`.
- **Generation:** Uses the `Engine` class from nanochat for streaming, KV-cached inference.

## 🚀 Key Commands
- **Train:** `bash runs/odinglish.sh`
- **Chat:** `python scripts/sovogpt_chat.py`
- **Eval:** `python scripts/sovogpt_eval.py`

## ⚠️ Important Nuances
- **No Weight Reuse:** Existing Llama weights are INCOMPATIBLE. The model must be trained from scratch.
- **MPS Priority:** Always ensure `torch.backends.mps.is_available()` is checked.
- **Transliteration:** All user input MUST be put through `to_roman_odia()` before tokenization.
