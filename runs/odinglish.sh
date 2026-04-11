#!/bin/bash
# ============================================================================
# Sovogpt × nanochat — Full Odinglish training pipeline for Mac M2 Pro (MPS)
# ============================================================================
#
# Usage:
#   cd sovogpt-main
#   bash runs/odinglish.sh
#
# This script:
#   1. Converts your ChatML data → nanochat JSONL
#   2. Trains a BPE tokenizer on the local Odinglish corpus
#   3. Pretrains a small GPT-2 (depth=6) from local Odinglish text
#   4. SFT on Odinglish conversations
#   5. Quick test prompt
# ============================================================================

set -e

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NANOCHAT_DIR="$PROJECT_ROOT/nanochat_engine"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

# Force float32 on MPS (MPS doesn't support bf16)
export NANOCHAT_DTYPE=float32

# wandb run name (set WANDB_RUN env var or defaults to "odinglish")
WANDB_RUN="${WANDB_RUN:-odinglish}"

echo "============================================"
echo "  Sovogpt × nanochat — Odinglish Pipeline"
echo "  Device: Apple M2 Pro (MPS GPU)"
echo "  Base dir: $NANOCHAT_BASE_DIR"
echo "============================================"

# ─── Step 0: Convert data ────────────────────────────────────────────────────
echo ""
echo ">>> Step 0: Converting ChatML → nanochat JSONL..."
cd "$PROJECT_ROOT"
python scripts/convert_to_nanochat.py --input agent_training_data.txt --output-dir data

# Copy the conversations file where nanochat expects it
cp data/odinglish_conversations.jsonl "$NANOCHAT_BASE_DIR/odinglish_conversations.jsonl"

# ─── Step 1: Activate nanochat venv ──────────────────────────────────────────
echo ""
echo ">>> Step 1: Activating nanochat environment..."
cd "$NANOCHAT_DIR"
if [ ! -x ".venv/bin/python" ]; then
    echo "ERROR: nanochat_engine/.venv is missing."
    echo "Run: cd nanochat_engine && uv venv && uv sync"
    exit 1
fi
source .venv/bin/activate
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "ERROR: nanochat dependencies are not installed in nanochat_engine/.venv"
    echo "Run: cd nanochat_engine && uv sync"
    exit 1
fi

# ─── Step 2: Train tokenizer on local corpus ─────────────────────────────────
echo ""
echo ">>> Step 2: Training BPE tokenizer on local Odinglish corpus..."
python -m scripts.tok_train \
    --text-corpus "$PROJECT_ROOT/data/odinglish_pretrain.txt" \
    --max-chars=2000000000

# ─── Step 4: Base pretrain (depth=6 GPT-2, ~30 min on M2 Pro) ───────────────
echo ""
echo ">>> Step 3: Base pretraining (depth=6, local corpus)..."
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=16 \
    --total-batch-size=8192 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=200 \
    --num-iterations=5000 \
    --text-corpus "$PROJECT_ROOT/data/odinglish_pretrain.txt" \
    --run=$WANDB_RUN

echo ""
echo ">>> Base eval (sampling only)..."
python -m scripts.base_eval --eval sample --device-batch-size=1 --max-per-task=16

# ─── Step 5: SFT on Odinglish conversations (~15 min) ───────────────────────
echo ""
echo ">>> Step 4: SFT on Odinglish conversations (~15 min)..."
python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=16 \
    --total-batch-size=8192 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --chatcore-every=-1 \
    --num-iterations=1500 \
    --custom-jsonl "$NANOCHAT_BASE_DIR/odinglish_conversations.jsonl" \
    --custom-jsonl-epochs=2 \
    --local-only \
    --run=$WANDB_RUN

# ─── Step 6: Quick smoke test ────────────────────────────────────────────────
echo ""
echo ">>> Step 5: Smoke test..."
python -m scripts.chat_cli -p "namaskar, tume kie?"

echo ""
echo "============================================"
echo "  ✅ Pipeline complete!"
echo "  Run: python scripts/sovogpt_chat.py"
echo "============================================"
