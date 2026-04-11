"""
Convert sovogpt ChatML training data → nanochat JSONL format.

Input:  agent_training_data.txt  (ChatML with <|im_start|>/<|im_end|>)
Output: data/odinglish_conversations.jsonl  (nanochat SFT format)
        data/odinglish_pretrain.txt          (plain text for tokenizer / base-pretrain)

Usage:
    python scripts/convert_to_nanochat.py [--input agent_training_data.txt]
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from indic_transliteration import sanscript

    HAS_TRANSLIT = True
except Exception:
    HAS_TRANSLIT = False

# ---------------------------------------------------------------------------
# Parsing helpers (reuse validation from prepare_odinglish_data.py)
# ---------------------------------------------------------------------------
BAD_MARKERS = ["<nooutput>", "<<search>>", "<<weather>>"]
ODIA_BLOCK_RE = re.compile(r"[\u0B00-\u0B7F]")
NON_ROMAN_RE = re.compile(r"[^a-z0-9 .,?!'/-]+")
WS_RE = re.compile(r"\s+")


def is_valid(text: str, max_len: int = 300) -> bool:
    low = text.lower()
    if any(m in low for m in BAD_MARKERS):
        return False
    if low.startswith("<<"):
        return False
    if len(text) < 2 or len(text) > max_len:
        return False
    return True


def to_roman_odia(text: str) -> str:
    cleaned = text.strip().replace("|", " ").replace("।", ".")
    if HAS_TRANSLIT and ODIA_BLOCK_RE.search(cleaned):
        try:
            cleaned = sanscript.transliterate(cleaned, sanscript.ORIYA, sanscript.ITRANS)
        except Exception:
            pass
    cleaned = cleaned.lower()
    cleaned = NON_ROMAN_RE.sub(" ", cleaned)
    cleaned = WS_RE.sub(" ", cleaned).strip()
    return cleaned


def parse_chatml(data: str):
    """Yield (user, assistant) pairs from ChatML data."""
    # Split on <|im_start|>system to get individual conversations
    conversations = [c.strip() for c in data.split("<|im_start|>system\n") if c.strip()]
    for conv in conversations:
        full = "<|im_start|>system\n" + conv
        # Extract user and assistant content
        user_match = re.search(
            r"<\|im_start\|>user\n(.*?)<\|im_end\|>", full, re.DOTALL
        )
        assistant_match = re.search(
            r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", full, re.DOTALL
        )
        if user_match and assistant_match:
            user = to_roman_odia(user_match.group(1))
            assistant = to_roman_odia(assistant_match.group(1))
            if is_valid(user) and is_valid(assistant):
                yield user, assistant


def main():
    parser = argparse.ArgumentParser(description="Convert ChatML → nanochat JSONL")
    parser.add_argument(
        "--input",
        default="agent_training_data.txt",
        help="Input ChatML file",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "odinglish_conversations.jsonl")
    pretrain_path = os.path.join(args.output_dir, "odinglish_pretrain.txt")

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        data = f.read()

    seen = set()
    count = 0
    pretrain_lines = []

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for user, assistant in parse_chatml(data):
            key = f"{user}||{assistant}"
            if key in seen:
                continue
            seen.add(key)

            # nanochat JSONL format: one JSON array per line
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
            jf.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

            # Collect text for pretraining corpus
            pretrain_lines.append(user)
            pretrain_lines.append(assistant)

    with open(pretrain_path, "w", encoding="utf-8") as pf:
        pf.write("\n".join(pretrain_lines))

    print(f"✅ Wrote {count} conversations → {jsonl_path}")
    print(f"✅ Wrote pretrain text ({len(pretrain_lines)} lines) → {pretrain_path}")


if __name__ == "__main__":
    main()
