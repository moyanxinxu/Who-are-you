from dataclasses import dataclass


@dataclass
class hp:
    name = "xxxx"
    author = "xxxx"
    data_path = "/path/to/identity.json"
    model_name = "/path/to/model/dir"
    bos_token = "<|im_start|>"

    r = 8
    lora_alpha = 32
    lora_dropout = 0.1

    output_dir = "./output/Qwen"
    lora_dir = "./output/lora"
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 2
    logging_steps = 10
    num_train_epochs = 10
    gradient_checkpointing = True
    save_steps = 100
    learning_rate = 1e-4
    save_on_each_node = True
