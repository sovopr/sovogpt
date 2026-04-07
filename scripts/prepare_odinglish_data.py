import argparse
import os
import re
from typing import Iterable, List, Sequence, Tuple

try:
    from indic_transliteration import sanscript

    HAS_TRANSLIT = True
except Exception:
    HAS_TRANSLIT = False


DEFAULT_INPUT_FILES = [
    "agent_training_data.txt",
    "real_chat_dataset.txt",
    "clean_chat_data.txt",
    "synthetic_chat.txt",
    "balanced_training.txt",
    "massive_training_data.txt",
    "final_mix.txt",
]

SYSTEM_INSTRUCTION = (
    "Instruction: You are an Odia AI assistant. Reply in natural Odia+English "
    "using English letters only (Roman script). Keep replies conversational."
)

ODIA_BLOCK_RE = re.compile(r"[\u0B00-\u0B7F]")
NON_ROMAN_RE = re.compile(r"[^a-z0-9 .,?!'/-]+")
WS_RE = re.compile(r"\s+")


SEED_PAIRS: Sequence[Tuple[str, str]] = [
    ("namaskar", "namaskar! kemiti achha?"),
    ("kemiti achha", "mu bhala achi, tume kemiti achha?"),
    ("tumara nama kana", "mora nama sovogpt."),
    ("tu kie", "mu sovogpt, tumara odia ai friend."),
    ("odia re katha heba", "han, ame odia english mix re katha haba."),
    ("mate help darkar", "sure, kana help darkar kuha."),
    ("aaji mora mood kharap", "chinta karana, mu achi, dhire dhire thik haba."),
    ("mu odia english re type karibi", "perfect, tume roman odia re type kara."),
    ("tume mora friend heba ki", "nischaya, mu tumara friend bhabe achi."),
    ("dhanyabad", "welcome! jebe darkar pacha."),
]


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


def parse_pairs_from_block(block: str) -> Tuple[str, str]:
    user = ""
    assistant = ""
    for line in block.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("user:") and not user:
            user = stripped.split(":", 1)[1].strip()
        elif low.startswith("sovogpt:") and not assistant:
            assistant = stripped.split(":", 1)[1].strip()
    return user, assistant


def parse_pairs(path: str) -> Iterable[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        data = handle.read()

    if "<|endoftext|>" in data:
        for block in data.split("<|endoftext|>"):
            user, assistant = parse_pairs_from_block(block)
            if user and assistant:
                yield user, assistant
        return

    lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
    for idx in range(len(lines) - 1):
        if lines[idx].lower().startswith("user:") and lines[idx + 1].lower().startswith("sovogpt:"):
            yield lines[idx].split(":", 1)[1].strip(), lines[idx + 1].split(":", 1)[1].strip()


def is_valid_pair(user: str, assistant: str) -> bool:
    if not user or not assistant:
        return False
    user_low = user.lower()
    assistant_low = assistant.lower()
    if "<nooutput>" in assistant_low:
        return False
    if "<<search>>" in assistant_low or "<<weather>>" in assistant_low:
        return False
    if assistant_low.startswith("<<"):
        return False
    if len(user) < 2 or len(user) > 140:
        return False
    if len(assistant) < 2 or len(assistant) > 220:
        return False
    return True


def build_dataset(input_files: Sequence[str], max_samples: int) -> List[str]:
    unique = set()
    rows: List[Tuple[str, str]] = []

    for src in input_files:
        if not os.path.exists(src):
            continue
        for raw_user, raw_assistant in parse_pairs(src):
            if not is_valid_pair(raw_user, raw_assistant):
                continue
            user = to_roman_odia(raw_user)
            assistant = to_roman_odia(raw_assistant)
            if not is_valid_pair(user, assistant):
                continue
            key = f"{user}||{assistant}"
            if key in unique:
                continue
            unique.add(key)
            rows.append((user, assistant))
            if len(rows) >= max_samples:
                break
        if len(rows) >= max_samples:
            break

    for user, assistant in SEED_PAIRS:
        key = f"{user}||{assistant}"
        if key in unique:
            continue
        unique.add(key)
        rows.append((user, assistant))

    output_rows = []
    for user, assistant in rows:
        output_rows.append(
            f"<|im_start|>system\n{SYSTEM_INSTRUCTION}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
        )
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create cleaned Odia-English Roman training data.")
    parser.add_argument("--output", default="agent_training_data.txt", help="Output dataset path")
    parser.add_argument("--max-samples", type=int, default=12000, help="Maximum unique pairs")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=DEFAULT_INPUT_FILES,
        help="Input dataset files to merge and clean",
    )
    args = parser.parse_args()

    records = build_dataset(args.inputs, args.max_samples)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.writelines(records)

    print(f"Wrote {len(records)} cleaned records to {args.output}")


if __name__ == "__main__":
    main()
