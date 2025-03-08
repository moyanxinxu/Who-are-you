from dataclasses import dataclass


@dataclass
class hp:
    name = "xxxx"
    author = "xxxxx"
    data_path = "/root/Documents/Who-are-you/src/data/identity.json"
    model_name = "Qwen/Qwen2.5-0.5B"
    bos_token = "<|im_start|>"

    r = 8
    lora_alpha = 32
    lora_dropout = 0.1

    output_dir = "./output/Qwen"
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 2
    logging_steps = 10
    num_train_epochs = 3
    gradient_checkpointing = True
    save_steps = 100
    learning_rate = 1e-4
    save_on_each_node = True
