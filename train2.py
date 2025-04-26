import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.abspath("llama-tuned2")
OUTPUT_PATH = "llama-tuned2"
CLEANED_DATA_PATH = "data/instructions_cleaned.json"

# STEP 1: Normalize HotpotQA-style JSON into Alpaca-style format
def clean_dataset(input_path, output_path):
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    cleaned = []
    for example in raw_data:
        instruction = str(example.get("question", ""))
        input_text = ""

        # Build context into input if present
        if 'context' in example and example['context']:
            context = example['context']
            titles = context.get("title", [])
            sentences = context.get("sentences", [])
            context_str = ""
            for title, sent_group in zip(titles, sentences):
                context_str += f"{title}: {' '.join(sent_group)}\n"
            input_text = context_str.strip()

        if 'source' in example and example['source']:
            input_text += f"\n\n(Source: {example['source']})"

        output = str(example.get("answer", ""))
        cleaned.append({
            "instruction": instruction,
            "input": input_text,
            "output": output
        })

    with open(output_path, "w") as f:
        json.dump(cleaned, f, indent=2)

# STEP 2: Format into instruction-style prompt
def format_dataset(example):
    prompt = f"### Instruction:\n{example['instruction']}\n"
    prompt += f"### Input:\n{example['input']}\n" if example.get("input") else "### Input:\n\n"
    prompt += f"### Response:\n{example['output']}"
    return {"text": prompt}

def main():
    # Clean raw dataset if needed
    clean_dataset("data/dev_data.json", CLEANED_DATA_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager"
    )
    logger.info("@train.py Model loaded succesfully")

    # Load cleaned dataset
    pre_dataset = load_dataset("json", data_files=CLEANED_DATA_PATH)["train"]
    dataset = pre_dataset.map(format_dataset)

    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
        
    tokenized_dataset = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        per_device_train_batch_size=12,
        num_train_epochs=3,
        gradient_accumulation_steps=1,
        learning_rate=0.0001,
        fp16=False,
        bf16=True,
        ddp_find_unused_parameters=False,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        report_to="none",
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Model Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
