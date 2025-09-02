#!/usr/bin/env python3
"""
train_completion_cpu_phase1_memsafe.py
CPU‑only LoRA fine‑tuning for Gemma 3 command completion
with context-aware <HIST> + <CURSOR> dataset from Hugging Face.
Optimized for low-RAM environments.
"""

import os
import torch
import numpy as np
import psutil
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
import transformers, accelerate

# === Debug: Environment versions ===
print("="*80)
print(f"[DEBUG] Transformers: {transformers.__version__} @ {transformers.__file__}")
print(f"[DEBUG] Accelerate: {accelerate.__version__} @ {accelerate.__file__}")
from accelerate import Accelerator
print(f"[DEBUG] parallelism_config present: {hasattr(Accelerator(), 'parallelism_config')}")
print("="*80)

# === Config ===
BASE_MODEL = "google/gemma-3-270m"
DATASET_REPO = "atekrugis/ssh_context_dataset"
MAX_LEN = 192  # reduced from 256 to save RAM
BATCH_SIZE = 2 # reduced from 4

def log_mem(stage):
    mem = psutil.virtual_memory()
    print(f"[MEM] {stage}: {mem.percent}% used ({mem.used/1e9:.2f} GB / {mem.total/1e9:.2f} GB)")

print("="*80)
print(f"[INFO] Starting CPU-only LoRA fine-tuning for {BASE_MODEL}")
print(f"[INFO] Using dataset from Hugging Face: {DATASET_REPO}")
print("="*80)

# === Tokenizer ===
print("[STEP] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"[INFO] Tokenizer loaded. EOS token id: {tokenizer.eos_token_id}")
log_mem("After tokenizer load")

# === Model ===
print("[STEP] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    device_map=None,
    attn_implementation="eager"
)
print(f"[INFO] Base model loaded: {type(model).__name__}")
log_mem("After model load")

# === LoRA Config ===
print("[STEP] Applying LoRA configuration...")
peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)
print("[INFO] LoRA applied.")
log_mem("After LoRA wrap")

# === Dataset Loading ===
print("[STEP] Loading dataset from Hugging Face...")
ds = load_dataset(
    DATASET_REPO,
    data_files={
        "train": "train_context.jsonl",
        "val": "val_context.jsonl"
    }
)
print(f"[INFO] Dataset loaded: {len(ds['train'])} train examples, {len(ds['val'])} val examples")
log_mem("After dataset load")

# === Tokenization Function ===
def tokenize(ex):
    prompt_ids = tokenizer(
        ex["prompt"], add_special_tokens=False,
        truncation=True, max_length=MAX_LEN
    )["input_ids"]

    comp_ids = tokenizer(
        ex["completion"], add_special_tokens=False,
        truncation=True, max_length=MAX_LEN
    )["input_ids"]

    input_ids = (prompt_ids + comp_ids)[:MAX_LEN]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + comp_ids
    labels = labels[:MAX_LEN]

    pad_len = MAX_LEN - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

print("[STEP] Tokenizing dataset...")
tok_ds = ds.map(tokenize, remove_columns=ds["train"].column_names)
print("[INFO] Tokenization complete.")
print(f"[DEBUG] Sample tokenized example (train[0]):\nPrompt: {ds['train'][0]['prompt']}\nCompletion: {ds['train'][0]['completion']}")
log_mem("After tokenization")

# === Metric Function ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    accuracy = (preds == labels)[mask].mean()
    return {"accuracy": accuracy}

# === Training Arguments ===
print("[STEP] Setting up training arguments...")
args = TrainingArguments(
    output_dir="out/gemma3_cc_cpu_phase1_memsafe",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,
    learning_rate=5e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=400,
    eval_strategy="steps",
    eval_steps=200,
    eval_accumulation_steps=1,
    weight_decay=0.01,
    optim="adamw_torch",
    report_to=[],
    dataloader_num_workers=0,
    load_best_model_at_end=False,  # save RAM
    metric_for_best_model="accuracy",
    greater_is_better=True
)
print("[INFO] Training arguments ready.")

# === Trainer ===
print("[STEP] Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["val"],
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
    compute_metrics=compute_metrics
)
print("[INFO] Trainer initialized.")
log_mem("After Trainer init")

# === Training ===
print("="*80)
print("[TRAINING] Starting training loop...")
print("="*80)
trainer.train()
log_mem("After training")
print("="*80)
print("[TRAINING] Training complete.")
print("="*80)

# === Save Model & Tokenizer ===
print("[STEP] Saving model and tokenizer...")
model.save_pretrained("out/gemma3_cc_cpu_phase1_memsafe_adapter")
tokenizer.save_pretrained("out/gemma3_cc_cpu_phase1_memsafe_adapter")
print("[INFO] Model and tokenizer saved.")

# === Quick test completions ===
print("[TEST] Generating sample completions from val set...")
model.eval()
for i in range(3):
    prompt = ds["val"][i]["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32)
    print(f"\n[Prompt]\n{prompt}")
    print(f"[Completion]\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")
