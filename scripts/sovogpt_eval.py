"""
Sovogpt 15-turn automated evaluation using nanochat engine.

Runs a scripted conversation and reports basic quality metrics.

Usage:
    cd sovogpt-main/nanochat_engine
    source .venv/bin/activate
    cd ..
    python scripts/sovogpt_eval.py
"""

import os
import sys
import re
import argparse

NANOCHAT_DIR = os.path.join(os.path.dirname(__file__), "..", "nanochat_engine")
sys.path.insert(0, NANOCHAT_DIR)

import torch

# ─── Transliteration ─────────────────────────────────────────────────────────
try:
    from indic_transliteration import sanscript
    HAS_TRANSLIT = True
except ImportError:
    HAS_TRANSLIT = False

ODIA_BLOCK_RE = re.compile(r"[\u0B00-\u0B7F]")
NON_ROMAN_RE = re.compile(r"[^a-z0-9 .,?!'/-]+")
WS_RE = re.compile(r"\s+")


def to_roman_odia(text: str) -> str:
    out = text.strip().replace("|", " ").replace("।", ".")
    if HAS_TRANSLIT and ODIA_BLOCK_RE.search(out):
        try:
            out = sanscript.transliterate(out, sanscript.ORIYA, sanscript.ITRANS)
        except Exception:
            pass
    out = out.lower()
    out = NON_ROMAN_RE.sub(" ", out)
    out = WS_RE.sub(" ", out).strip()
    return out


# ─── Evaluation prompts (same as test_eval.py) ───────────────────────────────
EVAL_PROMPTS = [
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
    "dhanyabad, tume bahut katha kahila. bye!",
]


def main():
    parser = argparse.ArgumentParser(description="Sovogpt 15-turn automated evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"], help="Evaluation device")
    parser.add_argument("--model-tag", type=str, default=None, help="Optional SFT checkpoint tag to load")
    parser.add_argument("--step", type=int, default=None, help="Optional checkpoint step to load")
    args = parser.parse_args()

    from nanochat.common import compute_init, autodetect_device_type
    from nanochat.engine import Engine
    from nanochat.checkpoint_manager import load_model

    print("=" * 60)
    print("  Sovogpt × nanochat — 15-Turn Evaluation")
    print("=" * 60)

    device_type = autodetect_device_type() if args.device == "auto" else args.device
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.step)

    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    engine = Engine(model, tokenizer)

    conversation_tokens = [bos]
    results = []

    for turn_num, prompt in enumerate(EVAL_PROMPTS, 1):
        user_text = to_roman_odia(prompt)

        # Add user turn
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_text))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)

        # Generate
        response_tokens = []
        for token_column, _ in engine.generate(
            conversation_tokens, num_samples=1, max_tokens=128, temperature=0.7, top_k=50
        ):
            token = token_column[0]
            response_tokens.append(token)

        # Decode response
        reply_text = tokenizer.decode(response_tokens).strip()

        # Ensure assistant_end
        if not response_tokens or response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)

        # ─── Quality checks ───────────────────────────────────────────
        is_empty = len(reply_text.strip()) == 0
        has_leak = any(marker in reply_text for marker in ["<|", "|>", "<|im_start|>", "<|im_end|>"])
        has_odia_unicode = bool(ODIA_BLOCK_RE.search(reply_text))

        status = "✅"
        issues = []
        if is_empty:
            issues.append("EMPTY")
            status = "❌"
        if has_leak:
            issues.append("TOKEN_LEAK")
            status = "⚠️"
        if has_odia_unicode:
            issues.append("UNICODE_ODIA")
            status = "⚠️"

        results.append({
            "turn": turn_num,
            "status": status,
            "issues": issues,
            "reply_len": len(reply_text),
        })

        print(f"\nTurn {turn_num} {status}")
        print(f"  You:     {prompt}")
        print(f"  Sovogpt: {reply_text}")
        if issues:
            print(f"  Issues:  {', '.join(issues)}")

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    ok = sum(1 for r in results if r["status"] == "✅")
    warn = sum(1 for r in results if r["status"] == "⚠️")
    fail = sum(1 for r in results if r["status"] == "❌")
    avg_len = sum(r["reply_len"] for r in results) / len(results)
    print(f"  ✅ Pass: {ok}  ⚠️ Warn: {warn}  ❌ Fail: {fail}")
    print(f"  Avg reply length: {avg_len:.0f} chars")
    print("=" * 60)


if __name__ == "__main__":
    main()
