#!/usr/bin/env python3
"""
Enhanced dataset builder for Gemma 3 command completion fine-tuning.
Sources:
- Hugging Face datasets (schema-aware mappers)
- TLDR pages from GitHub
- UCI Cybersecurity Shell Commands (local JSON)
- Optional Stack Exchange dumps (local XML)
"""

import json, os, re, random, hashlib, subprocess
from pathlib import Path
from datasets import load_dataset
import xml.etree.ElementTree as ET

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

def map_umer(row):
    steps = row.get("Solution Steps", "")
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

# === TLDR Pages from GitHub ===
def fetch_tldr():
    repo_url = "https://github.com/tldr-pages/tldr.git"
    local_dir = Path("data/tldr_repo")
    if not local_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(local_dir)])
    examples = []
    for md_file in local_dir.rglob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if line.strip().startswith("`") and line.strip().endswith("`"):
                cmd = line.strip("`\n")
                if safe(cmd):
                    sp = split_prefix_suffix(cmd)
                    if sp:
                        p, c = sp
                        examples.append({"prompt": p, "completion": c})
    return examples

# === UCI Cybersecurity Shell Commands ===
def fetch_uci_shell():
    path = Path("data/uci_shell.json")
    if not path.exists():
        print("UCI shell commands file not found, skipping.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = []
    for row in data:
        cmd = row.get("command") or row.get("cmd") or row.get("text")
        if cmd and safe(cmd):
            sp = split_prefix_suffix(cmd)
            if sp:
                p, c = sp
                examples.append({"prompt": p, "completion": c})
    return examples

# === Stack Exchange Dumps ===
def fetch_stackexchange(xml_path):
    if not Path(xml_path).exists():
        print(f"{xml_path} not found, skipping.")
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    examples = []
    for row in root.findall("row"):
        body = row.attrib.get("Body", "")
        m = re.search(r"<code>([a-z0-9_\-\.]+(?:\s+[^\n<]+)?)</code>", body, re.I)
        if m:
            cmd = m.group(1).strip()
            if safe(cmd):
                sp = split_prefix_suffix(cmd)
                if sp:
                    p, c = sp
                    examples.append({"prompt": p, "completion": c})
    return examples

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seen, items = set(), []

    # Hugging Face datasets
    hf_sources = [
        ("mecha-org/linux-command-dataset", map_mecha),
        ("sakkke/text-to-command-gemini", map_sakkke),
        ("Romit2004/LinuxCommands", map_romit),
        ("harpomaxx/unix-commands", map_harpomaxx),
        ("UmerSajid/IT-Troubleshooting-Dataset", map_umer),
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

    # TLDR Pages
    print("Fetching TLDR pages from GitHub…")
    for ex in fetch_tldr():
        k = dedup_key(ex["prompt"], ex["completion"])
        if k not in seen:
            seen.add(k)
            items.append(ex)

    # UCI Cybersecurity Shell Commands
    print("Fetching UCI Cybersecurity Shell Commands…")
    for ex in fetch_uci_shell():
        k = dedup_key(ex["prompt"], ex["completion"])
        if k not in seen:
            seen.add(k)
            items.append(ex)

    # Stack Exchange Dumps (optional)
    for dump in ["data/askubuntu.xml", "data/unix.xml", "data/serverfault.xml"]:
        print(f"Parsing {dump}…")
        for ex in fetch_stackexchange(dump):
            k = dedup_key(ex["prompt"], ex["completion"])
            if k not in seen:
                seen.add(k)
                items.append(ex)

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

    # Write out JSONL files
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/train.jsonl", "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(f"{OUT_DIR}/val.jsonl", "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Built completion corpus: {len(train)} train, {len(val)} val")

if __name__ == "__main__":
    main()