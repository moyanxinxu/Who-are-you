import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.config import hp

df = pd.read_json(hp.data_path, orient="columns")
ds = Dataset.from_pandas(df)


tokenizer = AutoTokenizer.from_pretrained(
    hp.model_name, bos_token=hp.bos_token, trust_remote_code=True
)
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(hp.model_name, trust_remote_code=True)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["c_attn", "c_proj", "w1", "w2"],
    inference_mode=False,
    r=hp.r,
    lora_alpha=hp.lora_alpha,
    lora_dropout=hp.lora_dropout,
)

def datapipe(example):
    instruction = example["instruction"]
    output = example["output"].format(name=hp.name, author=hp.author)

    instruction = [
        {"role": "user", "content": instruction},
    ]

    instruction = tokenizer.apply_chat_template(
        instruction, tokenize=True, add_generation_prompt=True
    )

    input_ids = instruction
    attention_mask = [1] * len(input_ids)
    label = [-100] * len(input_ids)

    output = f"{output}{tokenizer.eos_token}"
    output = tokenizer(output, add_special_tokens=False)

    input_ids += output["input_ids"]
    attention_mask += output["attention_mask"]
    label += output["input_ids"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}


ds = ds.map(datapipe, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir=hp.output_dir,
    per_device_train_batch_size=hp.per_device_train_batch_size,
    gradient_accumulation_steps=hp.gradient_accumulation_steps,
    logging_steps=hp.logging_steps,
    num_train_epochs=hp.num_train_epochs,
    gradient_checkpointing=hp.gradient_checkpointing,
    save_steps=hp.save_steps,
    learning_rate=hp.learning_rate,
    save_on_each_node=hp.save_on_each_node,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()

# model = model.merge_and_unload()
# model.save_pretrained(hp.lora_dir)
