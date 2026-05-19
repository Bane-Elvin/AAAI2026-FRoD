from datasets import load_dataset
from transformers import CLIPFeatureExtractor, CLIPForImageClassification, TrainingArguments, Trainer
from torch.optim import AdamW
import torch
import numpy as np
import evaluate
import os
import wandb
from peft import LoraConfig, VeraConfig, VBLoRAConfig, get_peft_model, FRODConfig
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import argparse



def get_lora_config(method: str, random_seed: int, sparse_rate: float):
    if method.lower() == "frod":
        return FRODConfig(

            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],

            save_projection=False,
            sparse_rate=sparse_rate,
            projection_prng_key=random_seed,
            bias="none",
            modules_to_save=["classifier"],
        )
    elif method.lower() == "vblora":
        return VBLoRAConfig(
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            num_vectors=90,
            vector_length=256,
            bias="none",
            save_only_topk_weights=True,
            modules_to_save=["classifier"],
        )
    elif method.lower() == "pissa":
        return LoraConfig(
            init_lora_weights="pissa_niter_16",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            bias="none",
            modules_to_save=["classifier"],
        )
    elif method.lower() == "randlora":
        return
    else:
        return LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def print_model_info(model):
    for name, param in model.named_parameters():
        print(name, param.numel())


def main(method: str, dataset_dir, device, args):
    dataset_name = dataset_dir.rsplit("/", 1)[-1]
    print(f"dataset_name: {dataset_name}")
    r_s = 3
    wandb.init(project=f'ablation_{dataset_name}_loss',
               name= f"{method}_S_{args.s}_l_{args.l_lr}_s_{args.s_lr}",
               mode="offline",
               dir= f"./wandb_result/{dataset_name}")

    if dataset_name == "svhn":
        ds = load_dataset(dataset_dir, 'cropped_digits')
    else:
        ds = load_dataset(dataset_dir)

    model_name_or_path = "/your/path/to/models/clip-vit-base-patch32"


    feature_extractor = CLIPFeatureExtractor.from_pretrained(model_name_or_path)


    def transform(example_batch):
        images = [x.convert("RGB") for x in example_batch['image']]
        inputs = feature_extractor(images, return_tensors='pt')
        inputs['labels'] = example_batch['label']
        return inputs

    prepared_ds = ds.with_transform(transform)

    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }

    train_dataloader = DataLoader(prepared_ds["train"], batch_size=64, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(prepared_ds["test"], batch_size=64, shuffle=False, collate_fn=collate_fn)

    labels = ds['train'].features['label'].names

    model = CLIPForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )
    if method.lower() != "fullft":
        config = get_lora_config(method, random_seed=r_s, sparse_rate=args.s)
        model = get_peft_model(model, config)

    print_trainable_parameters(model)


    if method.lower() == "frod":
        optimizer = AdamW(
            [
                {"params": [p for n, p in model.named_parameters() if "FROD_lambda_S" in n], "lr": args.s_lr},
                {"params": [p for n, p in model.named_parameters() if "FROD_lambda_l" in n], "lr": args.l_lr},
                {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": 1e-4},
            ]
        )
    elif method.lower() == "fullft":
        optimizer = AdamW(model.parameters(), lr=1e-4)

    best_accuracy = 0.0
    total_steps = 0
    model.to(device)
    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader)
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 10 == 0:
                progress_bar.set_description(f"Loss: {loss.item():.4f}")
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                }, step=total_steps)
                total_steps += 1

        print(f"Train loss: {total_loss / len(train_dataloader):.4f}")


        metric = evaluate.load("accuracy")
        model.eval()
        eval_start = time.time()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        eval_time = time.time() - eval_start
        eval_metric = metric.compute()
        print(f"Eval Accuracy: {eval_metric['accuracy']:.4f}")
        wandb.log({
            "eval/loss": None,
            "eval/accuracy": eval_metric['accuracy'],
            "eval/runtime": eval_time,
            "eval/steps_per_second": len(eval_dataloader) / eval_time,
            "eval/samples_per_second": len(prepared_ds["test"]) / eval_time
        }, step=total_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple models with specified dataset and device")
    parser.add_argument(
        "--model",
        type=str,
        default="FROD",
        help="Name of the model to run (default: vera)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/your/path/to/datasets/stanford_cars",
        help="Path to the dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use: cpu or cuda"
    )
    parser.add_argument(
        "--l_lr",
        type=float,
        default=5e-4,
        help="learning rate of \Sigma_i (default: 5e-4)"
    )
    parser.add_argument(
        "--s_lr",
        type=float,
        default=5e-5,
        help="learning rate of S (default: 5e-5)"
    )
    parser.add_argument(
        "--s",
        type=float,
        default=0.02,
        help="sparsity rate of S (default: 0.02)"
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    main(args.model, args.dataset, device, args)
