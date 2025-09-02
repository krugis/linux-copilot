#!/usr/bin/env python3
"""
augment_context_dataset.py
Builds context-aware training data for SSH completion assistant.
"""

import json
import random
from pathlib import Path
import re
import hashlib

# === Config ===
DATA_DIR = Path("data/cc")
HIST_LEN = 3           # number of previous commands to include
MAX_LEN = 128          # max tokens (handled later in tokenizer)
SEED = 42
random.seed(SEED)

# === Safety filter ===
DANGEROUS = [
    r"\brm\s+-rf\s+/\b",
    r"\bmkfs(\.\w+)?\s+/dev/\w+\b",
    r"\bdd\s+if=/dev/zero\s+of=/dev/\w+\b",
    r":\(\)\s*\{\s*:\|\:&\s*;\s*\}\s*:",  # fork bomb
    r"\bshutdown\s+-h\s+now\b",
    r"\breboot\b",
]
DANGER = [re.compile(p, re.I) for p in DANGEROUS]

def safe(cmd: str) -> bool:
    return not any(p.search(cmd) for p in DANGER)

def dedup_key(p, c):
    return hashlib.md5((p.strip()+"|||"+c.strip()).encode()).hexdigest()

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def augment_split(split_name):
    src_path = DATA_DIR / f"{split_name}.jsonl"
    out_path = DATA_DIR / f"{split_name}_context.jsonl"
    rows = load_jsonl(src_path)
    print(f"[INFO] Loaded {len(rows)} {split_name} examples.")

    augmented = []
    seen = set()

    for i in range(len(rows)):
        # Build history from previous HIST_LEN commands
        hist_indices = list(range(max(0, i-HIST_LEN), i))
        history_cmds = [rows[j]["prompt"] + rows[j]["completion"] for j in hist_indices]

        # Current command
        full_cmd = rows[i]["prompt"] + rows[i]["completion"]
        if not safe(full_cmd):
            continue

        toks = full_cmd.strip().split()
        if len(toks) < 2:
            continue

        # Random truncation point (simulate partial typing)
        cut = random.randint(1, len(toks)-1)
        partial = " ".join(toks[:cut])
        rest = " " + " ".join(toks[cut:])

        prompt = "<HIST>\n" + "\n".join(history_cmds) + "\n<CURSOR>\n" + partial
        completion = rest

        k = dedup_key(prompt, completion)
        if k in seen:
            continue
        seen.add(k)

        augmented.append({"prompt": prompt, "completion": completion})

    save_jsonl(augmented, out_path)
    print(f"[INFO] Wrote {len(augmented)} context-aware examples to {out_path}")

def main():
    for split in ["train", "val"]:
        augment_split(split)

if __name__ == "__main__":
    main()
