import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import config
from datasets import Dataset
import os
from transformers import TFAutoModelForCausalLM
from transformers import create_optimizer, AdamWeightDecay
from tensorflow.keras.callbacks import TensorBoard
import math


training_file = Path(__file__).parent / config.FILE_PATH.get("training_data")
tokenizer_path = Path(__file__).parent / config.FILE_PATH.get("tokenizer_path")
tensorboard_log_dir = Path(__file__).parent / config.FILE_PATH.get(
    "tensorboard_log_dir"
)
model_name = config.MODEL.get("name")
model_save_path = Path(__file__).parent / config.MODEL.get("checkpoint")


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == 128:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, though you could add padding instead if the model supports it
    # In this, as in all things, we advise you to follow your heart
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


if __name__ == "__main__":
    df = pd.read_pickle(training_file)
    if len(os.listdir(tokenizer_path)) == 0:
        print("Preparing tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_path)

    dataset = Dataset.from_pandas(pd.DataFrame(df), preserve_index=False)
    dataset_dict = dataset.train_test_split(test_size=0.1, seed=2023)
    print("Preparing dataset")
    train_val_dataset = dataset_dict["train"]
    # test_dataset = dataset_dict['test']
    train_val_dataset_dict = train_val_dataset.train_test_split(
        test_size=0.1, seed=2023
    )
    train_dataset = train_val_dataset_dict["train"]
    val_dataset = train_val_dataset_dict["test"]
    print("Tokenizing dataset")
    train_dataset = train_dataset.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )

    val_dataset = val_dataset.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )
    block_size = 128
    print("Grouping text")
    lm_train_datasets = train_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    lm_val_datasets = val_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    print("Loading the model")
    model = TFAutoModelForCausalLM.from_pretrained(model_name)
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
    model.compile(optimizer=optimizer, jit_compile=True)
    train_set = model.prepare_tf_dataset(
        lm_train_datasets,
        shuffle=True,
        batch_size=16,
    )

    validation_set = model.prepare_tf_dataset(
        lm_val_datasets,
        shuffle=False,
        batch_size=16,
    )
    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir)
    print("Model training")
    model.fit(
        train_set,
        validation_data=validation_set,
        epochs=5,
        callbacks=[tensorboard_callback],
    )
    eval_loss = model.evaluate(validation_set)
    print(f"Perplexity: {math.exp(eval_loss):.2f}")
    print("Saving the model")
    model.save_pretrained(model_save_path)
