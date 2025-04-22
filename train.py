import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.abspath("model")
OUTPUT_PATH = "llama-tuned"

def format_dataset(example):
    prompt = f"### Instruction:\n{example['instruction']}\n"
    if example.get("input"):
        prompt += f"### Input:\n{example['input']}\n"
    prompt += f"### Response:\n{example['output']}"
    return {"text": prompt}

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager"
    )
    logger.info("@train.py Model loaded succesfully")

    #Load dataset
    pre_dataset = load_dataset("json", data_files="data/instructions.json")["train"]
    dataset = pre_dataset.map(format_dataset)

    def tokenize(example):
        tokens= tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
        
    tokenized_dataset = dataset.map(tokenize, batched=True)

    #TRaining args
    training_args= TrainingArguments(
        output_dir=OUTPUT_PATH,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        gradient_accumulation_steps=4,
        learning_rate=0.0001,
        fp16=False,
        bf16=True,
        save_steps=200,
        save_total_limit=2,
        logging_dir="./logs/application.log",
        report_to="none",
        optim="adamw_torch"
    )

    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    #TRain
    trainer.train()

    #Save
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Model Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

    
