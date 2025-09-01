#!/usr/bin/env python3
import os, json, re, random, hashlib
from pathlib import Path

# === Config ===
UCI_ROOT = Path("data/data_in/uci_data")  # path to your UCI dataset root
OUT_FILE = Path("data/uci_shell.jsonl")
SEED = 42
random.seed(SEED)

# Dangerous command patterns to skip
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

def split_prefix_suffix(cmd: str):
    toks = cmd.strip().split()
    if len(toks) < 2:
        return None
    cut = random.randint(1, len(toks)-1)
    prefix = " ".join(toks[:cut])
    suffix = " " + " ".join(toks[cut:])
    return prefix, suffix

def dedup_key(p, c):
    return hashlib.md5((p.strip()+"|||"+c.strip()).encode()).hexdigest()

def main():
    seen = set()
    count = 0
    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for root, _, files in os.walk(UCI_ROOT):
            for fname in files:
                fpath = Path(root) / fname
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or not line.startswith("{"):
                                continue
                            try:
                                rec = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            cmd = rec.get("cmd")
                            if not cmd or not safe(cmd):
                                continue
                            sp = split_prefix_suffix(cmd)
                            if not sp:
                                continue
                            p, c = sp
                            k = dedup_key(p, c)
                            if k in seen:
                                continue
                            seen.add(k)
                            out.write(json.dumps({"prompt": p, "completion": c}, ensure_ascii=False) + "\n")
                            count += 1
                except (UnicodeDecodeError, OSError):
                    continue
    print(f"Wrote {count} examples to {OUT_FILE}")

if __name__ == "__main__":
    main()
