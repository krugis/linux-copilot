#!/usr/bin/env python3
"""
train_completion_cpu.py
CPU‑only LoRA fine‑tuning for Gemma 3 command completion.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model

BASE_MODEL = "google/gemma-3-270m"
DATA_DIR = "data/cc"
MAX_LEN = 128

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map=None
)

peft_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)

ds = load_dataset("json", data_files={"train": f"{DATA_DIR}/train.jsonl", "val": f"{DATA_DIR}/val.jsonl"})

def tokenize(ex):
    prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
    comp_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + comp_ids
    labels = [-100]*len(prompt_ids) + comp_ids
    return {
        "input_ids": input_ids[:MAX_LEN],
        "attention_mask": [1]*min(len(input_ids), MAX_LEN),
        "labels": labels[:MAX_LEN]
    }

tok_ds = ds.map(tokenize, remove_columns=ds["train"].column_names)

args = TrainingArguments(
    output_dir="out/gemma3_cc_cpu",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=200,
    weight_decay=0.01,
    optim="adamw_torch",
    report_to=[],
    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["val"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
)

trainer.train()
model.save_pretrained("out/gemma3_cc_cpu_adapter")
tokenizer.save_pretrained("out/gemma3_cc_cpu_adapter")
