#!/usr/bin/env python3
"""
Dataset-aware builder for Gemma 3 command completion fine-tuning.
Pulls from Hugging Face + Kaggle, maps each dataset according to its schema,
filters unsafe commands, generates prefix→suffix pairs, deduplicates, and outputs train/val JSONL.
"""

import json, os, re, random, hashlib
from datasets import load_dataset

OUT_DIR = "data/cc"
VAL_SPLIT = 0.02
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

def safe(s: str) -> bool:
    return not any(p.search(s) for p in DANGER)

def dedup_key(p, c):
    return hashlib.md5((p.strip()+"|||"+c.strip()).encode()).hexdigest()

def split_prefix_suffix(cmd: str):
    toks = cmd.strip().split()
    if len(toks) < 2:
        return None
    cut = random.randint(1, len(toks)-1)
    prefix = " ".join(toks[:cut])
    suffix = " " + " ".join(toks[cut:])
    return prefix, suffix

# === Dataset-specific mappers ===

def map_mecha(row):
    cmd = row.get("output")
    if not cmd: return None
    cmd = cmd.strip()
    if not safe(cmd): return None
    sp = split_prefix_suffix(cmd)
    if not sp: return None
    p, c = sp
    return {"prompt": p, "completion": c}

def map_sakkke(row):
    cmd = row.get("output")
    if not cmd: return None
    cmd = cmd.strip()
    if not safe(cmd): return None
    sp = split_prefix_suffix(cmd)
    if not sp: return None
    p, c = sp
    return {"prompt": p, "completion": c}

def map_romit(row):
    cmd = row.get("command")
    if not cmd: return None
    cmd = cmd.strip()
    if not safe(cmd): return None
    sp = split_prefix_suffix(cmd)
    if not sp: return None
    p, c = sp
    return {"prompt": p, "completion": c}

def map_harpomaxx(row):
    cmd = row.get("output")
    if not cmd: return None
    cmd = cmd.strip()
    if not safe(cmd): return None
    sp = split_prefix_suffix(cmd)
    if not sp: return None
    p, c = sp
    return {"prompt": p, "completion": c}

def map_tldr(row):
    examples = row.get("examples")
    if not examples: return None
    cmd = examples[0].get("command")
    if not cmd: return None
    cmd = cmd.strip()
    if not safe(cmd): return None
    sp = split_prefix_suffix(cmd)
    if not sp: return None
    p, c = sp
    return {"prompt": p, "completion": c}

def map_bash_help(row):
    cmd = row.get("command")
    if not cmd: return None
    cmd = cmd.strip()
    if not safe(cmd): return None
    sp = split_prefix_suffix(cmd)
    if not sp: return None
    p, c = sp
    return {"prompt": p, "completion": c}

def map_umer(row):
    steps = row.get("Solution Steps", "")
    # crude command detection
    if re.search(r"\b(ls|cd|cat|grep|chmod|chown|systemctl|docker|kubectl|apt|yum|ssh)\b", steps):
        cmd_match = re.search(r"([a-z0-9_\-\.]+(?:\s+[^\n`]+)?)", steps)
        if cmd_match:
            cmd = cmd_match.group(0).strip()
            if not safe(cmd): return None
            sp = split_prefix_suffix(cmd)
            if not sp: return None
            p, c = sp
            return {"prompt": p, "completion": c}
    return None

def map_askubuntu(row):
    ans = row.get("accepted_answer") or ""
    cmd_match = re.search(r"^([a-z0-9_\-\.]+(?:\s+[^\n`]+)?)", ans, re.M)
    if cmd_match:
        cmd = cmd_match.group(0).strip()
        if not safe(cmd): return None
        sp = split_prefix_suffix(cmd)
        if not sp: return None
        p, c = sp
        return {"prompt": p, "completion": c}
    return None

# === Main builder ===

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seen, items = set(), []

    hf_sources = [
        ("mecha-org/linux-command-dataset", map_mecha),
        ("sakkke/text-to-command-gemini", map_sakkke),
        ("Romit2004/LinuxCommands", map_romit),
        ("harpomaxx/unix-commands", map_harpomaxx),
        ("tldr-pages/tldr", map_tldr),
        ("Abirate/bash-command-help", map_bash_help),
        ("UmerSajid/IT-Troubleshooting-Dataset", map_umer),
        ("flax-sentence-embeddings/askubuntu", map_askubuntu),
    ]

    for name, mapper in hf_sources:
        try:
            print(f"Loading {name}…")
            ds = load_dataset(name, split="train")
            for row in ds:
                ex = mapper(row)
                if not ex: continue
                k = dedup_key(ex["prompt"], ex["completion"])
                if k in seen: continue
                seen.add(k)
                items.append(ex)
        except Exception as e:
            print(f"Skipping {name}: {e}")

    # Optional: Kaggle integration here if desired

    # Synthetic augmentation
    COMMON_CMDS = [
        "tar -czvf backup.tar.gz /path/to/data",
        "ssh user@server.example.com",
        "docker run -it ubuntu:22.04 bash",
        "kubectl get pods -n kube-system",
        "apt install nginx",
    ]
    for cmd in COMMON_CMDS:
        sp = split_prefix_suffix(cmd)
        if sp:
            p, c = sp
            k = dedup_key(p, c)
            if k not in seen:
                seen.add(k)
                items.append({"prompt": p, "completion": c})

    random.shuffle(items)

    # Train/val split
    n = len(items)
    val_n = max(1000, int(VAL_SPLIT * n))
    train, val = items[val_n:], items[:val_n]

    with open(f"{OUT_DIR}/train.jsonl", "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(f"{OUT_DIR}/val.jsonl", "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Built completion corpus: {len(train)} train, {len(val)} val")

if __name__ == "__main__":
    main()
