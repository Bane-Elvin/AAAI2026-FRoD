import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import time
import argparse
from tqdm import tqdm

import wandb
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
    set_seed,
)

from peft import get_peft_model, FRODConfig, VeraConfig


def get_peft_config(method: str, random_seed: int):
    if method.lower() == "frod":
        return FRODConfig(
            task_type="SEQ_CLS",
            target_modules=["query", "key", "value", "output.dense", "intermediate.dense"],
            save_projection=False,
            sparse_rate=0.02,
            projection_prng_key=random_seed,
            modules_to_save=["classifier"],
        )
    if method.lower() == "vera":
        return  VeraConfig(
            task_type="SEQ_CLS",
            r=256,
            target_modules=["query", "key", "value","output.dense","intermediate.dense"],
            save_projection=False,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} | All params: {all_params} | Trainable %: {100 * trainable_params / all_params:.2f}%")


def collate_fn(examples, tokenizer):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


def main(method: str, task: str, device, model_name_or_path: str, dataset_root: str):
    random_seed = 3
    wandb.init(project=f'glue_{task}_loss', name=method)

    batch_size = 64
    max_length = 256
    num_epochs = 5

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset(dataset_root, task)
    metric = evaluate.load("glue", "sst2")

    def tokenize_function(examples):
        if task in ["sst2", "cola"]:
            return tokenizer(examples["sentence"], truncation=True, max_length=max_length)
        elif task in ["mrpc", "rte"]:
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
        elif task == "qnli":
            return tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=max_length)
        elif task == "stsb":
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
        elif task == "qqp":
            return tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=max_length)
        elif task == "mnli":
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=max_length)

    if task in ["sst2", "cola"]:
        tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
    elif task in ["mrpc", "rte"]:
        tokenized_datasets = datasets.map(tokenize_function, batched=True,
                                          remove_columns=["idx", "sentence1", "sentence2"])
    elif task == "qnli":
        tokenized_datasets = datasets.map(tokenize_function, batched=True,
                                          remove_columns=["idx", "question", "sentence"])
    elif task == "stsb":
        tokenized_datasets = datasets.map(tokenize_function, batched=True,
                                          remove_columns=["idx", "sentence1", "sentence2"])
    elif task == "qqp":
        tokenized_datasets = datasets.map(tokenize_function, batched=True,
                                          remove_columns=["idx", "question1", "question2"])
    elif task == "mnli":
        tokenized_datasets = datasets.map(tokenize_function, batched=True,
                                          remove_columns=["idx", "premise", "hypothesis"])

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    print(tokenized_datasets["train"][:1])
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda x: collate_fn(x, tokenizer))
    if task in ["mnli"]:
        eval_dataloader = DataLoader(tokenized_datasets["validation_matched"], batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda x: collate_fn(x, tokenizer))
    else:
        eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda x: collate_fn(x, tokenizer))

    if task == "stsb":
        num_labels = 1
    elif task == "mnli":
        num_labels = 3
    else:
        num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)

    if task == "stsb":
        model.config.problem_type = "regression"

    if method != "fullft":
        peft_config = get_peft_config(method, random_seed)
        model = get_peft_model(model, peft_config)

    # print(model)
    print_trainable_parameters(model)
    if method == "frod":
        optimizer = AdamW([
            {"params": [p for n, p in model.named_parameters() if "FROD_lambda_S" in n], "lr": 5e-5},
            {"params": [p for n, p in model.named_parameters() if "FROD_lambda_l" in n], "lr": 5e-4},
            {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": 1e-5},
        ])
    elif method == "vera":
        optimizer = AdamW(
            [
                {"params": [p for n, p in model.named_parameters() if "vera_lambda_" in n], "lr": 1e-3},
                {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": 1e-5},
            ]
        )
    elif method == "fullft":
        optimizer = AdamW(model.parameters(), lr=1e-5)

    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    model.to(device)
    best_metric = -float("inf")
    total_steps = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 10 == 0:
                tqdm.write(f"Step {step} | Loss: {loss.item():.4f}")
                wandb.log({"train/loss": loss.item(),
                           "train/learning_rate": optimizer.param_groups[0]['lr'],
                           }, step=total_steps)
                total_steps += 1
        print(f"Train loss: {total_loss / len(train_dataloader):.4f}")

        # Eval
        model.eval()
        eval_start = time.time()
        for batch in tqdm(eval_dataloader):


            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            if task == "stsb":
                preds = outputs.logits.squeeze()
            else:
                preds = outputs.logits.argmax(dim=-1)
            metric.add_batch(predictions=preds, references=batch["labels"])

        eval_metric = metric.compute()
        print(f"Eval Metric: {eval_metric}")
        eval_time = time.time() - eval_start
        # if task == "stsb":
        #     current_metric = eval_metric['pearson']
        # elif task == "cola":
        #     current_metric = eval_metric['accuracy']
        # else:
        #     current_metric = eval_metric['accuracy']
        wandb.log({
            "eval/loss": None,
            "eval/accuracy": eval_metric,
            "eval/runtime": eval_time,
            "eval/steps_per_second": len(eval_dataloader) / eval_time,
        }, step=total_steps)



        if eval_metric["accuracy"] > best_metric:
            best_metric = eval_metric["accuracy"]
            model.save_pretrained(f"./result/{method}_{task}")
            print("Best model saved.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="fullft", help="PEFT method, e.g., frod")
    parser.add_argument("--task", type=str, default="sst2", help="GLUE task, e.g., sst2")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/your/path/to/models/roberta-large",
        help="Path to the RoBERTa model"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/your/path/to/datasets/glue",
        help="Path to the GLUE dataset root"
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    main(args.method, args.task, device, args.model_path, args.dataset_root)
