import os
import torch
from torch.utils.data import DataLoader
import evaluate
import wandb
import argparse
from tqdm import tqdm
import time
from datasets import load_dataset, DatasetDict
from transformers import LlamaTokenizer, LlamaForSequenceClassification, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig
from torch.cuda.amp import autocast, GradScaler

# --- PEFT Configuration ---
def get_lora_config(method: str, random_seed: int = 3):
    """
    Returns the PEFT configuration based on the specified method.
    """
    if method.lower() == "FROD":
        from peft import FRODConfig
        return FRODConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
            save_projection=False,
            sparse_rate=0.02,
            projection_prng_key=random_seed,
            bias="none",
            modules_to_save=["score"],  # LlamaForSequenceClassification uses 'score' as the classifier head name
        )
    # Add other PEFT methods like VBLoRA, PiSSA if needed, following the CLIP script's pattern
    # elif method.lower() == "vblora": ...
    else:  # Default to LoRA
        return LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["score"],  # LlamaForSequenceClassification uses 'score' as the classifier head name
        )


# --- Utility Functions ---
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


# --- Data-specific Prompting and Labeling ---
# These functions are specific to the text classification tasks
def build_prompt(example, task):
    if task == "piqa_preop":
        return f"Physical situation description:\n{example['goal']}\nChoice A:\n{example['sol1']}\nChoice B:\n{example['sol2']}\nQuestion:\nWhich solution is more physically plausible, Choice A or Choice B?"
    elif task == "siqa":
        return f"Please choose the correct Choice to the question: {example['context']}, Question:\n{example['question']}\nChoice A:\n{example['answerA']}\nChoice B:\n{example['answerB']}\nChoice C:\n{example['answerC']}\nWhich choice best answers the question, Choice A, B, or C?"
    elif task == "hellaswag":
        endings = "".join([f"Ending {chr(65 + i)}:\n{example['endings'][i]}\n" for i in range(len(example['endings']))])
        return f"Context:\n{example['ctx']}\n{endings}Question:\nWhich ending is the most plausible continuation, Ending A, B, C, or D?"
    elif task == "winogrande":
        return f"Please choose the correct answer to fill in the blank to complete the given sentence: {example['sentence']}\n\nOption1: {example['option1']}\nOption2: {example['option2']}."
    elif task in ["arc_easy", "arc_challenge"]:
        if 'choices' in example and 'label' in example['choices'] and 'text' in example['choices']:
            opts = "".join([f"Option {example['choices']['label'][i]}:\n{example['choices']['text'][i]}\n" for i in
                            range(len(example['choices']['text']))])
            return f"Science question:\n{example['question']}\n{opts}Question:\nWhich option is the correct answer?"
        else:
            print(f"Warning: Missing 'choices' or 'label'/'text' in example for ARC tasks: {example.keys()}")
            return ""
    elif task == "openbookqa":
        if 'choices' in example and 'label' in example['choices'] and 'text' in example['choices']:
            opts = "".join([f"Option {example['choices']['label'][i]}:\n{example['choices']['text'][i]}\n" for i in
                            range(len(example['choices']['text']))])
            return f"Please choose the correct answer to the question:{example['question_stem']}\n{opts}."
        else:
            print(f"Warning: Missing 'choices' or 'label'/'text' in example for OpenBookQA: {example.keys()}")
            return ""
    elif task == "boolq":
        return f"Reference document:\n{example['passage']}\nQuestion:\n{example['question']}"
    else:
        raise NotImplementedError(f"Prompt for dataset {task} not implemented.")


def label_to_idx(example, task, num_labels):
    try:
        if task == "piqa_preop":
            label = int(example["label"])
        elif task == "siqa":
            label = int(example["label"])
            if label >= 1 and label <= num_labels:
                label -= 1
            else:
                print(f"Warning: Original SIQA label {label} is out of expected 1-based range [1, {num_labels}].")
                return -1

        elif task == "boolq":
            label = int(example["answer"])
        elif task == "winogrande":
            label = int(example["answer"]) - 1
        elif task in ["arc_easy", "arc_challenge", "openbookqa"]:
            if "answerKey" in example:
                if isinstance(example["answerKey"], str) and example["answerKey"].isalpha():
                    label = ord(example["answerKey"].upper()) - ord('A')
                elif isinstance(example["answerKey"], (int, str)):
                    label = int(example["answerKey"]) - 1
                else:
                    print(
                        f"Warning: Unexpected 'answerKey' type for {task}: {type(example['answerKey'])}. Example: {example}")
                    return -1
            else:
                print(f"Warning: Missing 'answerKey' for {task}. Example: {example.keys()}")
                return -1
        else:
            raise NotImplementedError(f"Label for dataset {task} not implemented.")


        if 0 <= label < num_labels:
            return label
        else:
            print(
                f"Warning: Final label value {label} for dataset {task} with {num_labels} labels is out of 0-based range [0, {num_labels - 1}]. Example keys: {example.keys()}")
            return -1
    except (KeyError, ValueError) as e:
        print(f"Error processing label for dataset {task}. Exception: {e}. Example keys: {example.keys()}")
        return -1


# --- Main Training and Evaluation Function ---
def main(method: str, dataset_name: str, device: torch.device, model_path: str):
    num_epochs = 6
    batch_size = 4
    gradient_accumulation_steps = 8

    # Mapping of dataset names to their number of labels
    num_label_dict = {
        "piqa_preop": 2, "siqa": 3, "hellaswag": 4, "winogrande": 2,
        "arc_easy": 4, "arc_challenge": 4, "openbookqa": 4, "boolq": 2
    }
    num_labels = num_label_dict[dataset_name]
    print(f"Dataset: {dataset_name}, Number of labels: {num_labels}")
    print(f"Current physical batch size: {batch_size}, Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

    wandb.init(project=f'llama_{dataset_name}_classification', name=method)

    # 1. Load Tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token
        print(f"Tokenizer pad_token set to eos_token: {tokenizer.pad_token}")

    # 2. Load and Preprocess Dataset
    print(f"Attempting to load dataset: /your/path/to/datasets/{dataset_name}")
    try:
        if dataset_name == "winogrande":
            raw_ds = load_dataset(f"/your/path/to/datasets/{dataset_name}", "winogrande_l")
        elif dataset_name == "arc_challenge":
            raw_ds = load_dataset(f"/your/path/to/datasets/ai2_arc", "ARC-Challenge")
        elif dataset_name == "arc_easy":
            raw_ds = load_dataset(f"/your/path/to/datasets/ai2_arc", "ARC-Easy")
        else:
            raw_ds = load_dataset(f"/your/path/to/datasets/{dataset_name}")
        print(f"Dataset '{dataset_name}' loaded successfully. Splits: {raw_ds.keys()}")
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        return

    def transform(example_batch):
        prompts, labels = [], []


        if not example_batch or not any(example_batch.values()):
            print("Warning: Received an empty or malformed example_batch in transform. Returning empty dictionary.")
            return {}

        num_examples = len(next(iter(example_batch.values())))

        for i in range(num_examples):
            example = {key: val[i] for key, val in example_batch.items()}

            label = label_to_idx(example, dataset_name, num_labels)

            if label != -1:
                prompt = build_prompt(example, dataset_name)
                # print(f"Example prompt: {prompt}")
                if prompt:
                    prompts.append(prompt)
                    labels.append(label)
                else:
                    print(
                        f"Warning: Empty prompt generated for an example in dataset {dataset_name}. Skipping this example.")
            else:
                print(
                    f"Warning: Label filtering (label = {label}) for an example in dataset {dataset_name}. Skipping this example.")

        if not prompts:
            print(
                f"Warning: After processing {num_examples} examples, 'prompts' list is empty. This batch will return an empty dictionary.")
            return {}

        inputs = tokenizer(prompts, truncation=True, max_length=512, padding="max_length", return_tensors='pt')
        inputs['labels'] = torch.tensor(labels, dtype=torch.long)

        return inputs

    prepared_ds = DatasetDict()
    for split in raw_ds.keys():
        prepared_ds[split] = raw_ds[split].with_transform(transform)
    print(f"Dataset transformed successfully. Prepared splits: {prepared_ds.keys()}")


    def collate_fn(batch):
        filtered_batch = [item for item in batch if item]
        if not filtered_batch:
            return {}

        collated = {
            'input_ids': torch.stack([x['input_ids'].squeeze() for x in filtered_batch]),
            'attention_mask': torch.stack([x['attention_mask'].squeeze() for x in filtered_batch]),
            'labels': torch.stack([x['labels'].squeeze() for x in filtered_batch])
        }
        return collated

    train_dataloader = DataLoader(prepared_ds["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_split = "validation" if "validation" in prepared_ds else "test"
    print(f"Using '{eval_split}' split for evaluation.")
    eval_dataloader = DataLoader(prepared_ds[eval_split], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Load Model
    print(f"Loading LlamaForSequenceClassification model from {model_path} with {num_labels} labels.")
    model = LlamaForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        # torch_dtype=torch.float16,
        device_map="auto",  # Automatically handle device placement for large models
    )
    print("Model loaded.")

    if method.lower() != "fullft":
        config = get_lora_config(method)
        model = get_peft_model(model, config)
        print(f"PEFT model (method: {method}) applied.")

    print_trainable_parameters(model)
    model.config.pad_token_id = tokenizer.pad_token_id  # Ensure model pad token id is set
    print(f"Model pad_token_id set to {model.config.pad_token_id}")

    # 5. Setup Optimizer and Scheduler
    if method.lower() == "FROD":
        optimizer = AdamW([
            {"params": [p for n, p in model.named_parameters() if "FROD_lambda_S" in n], "lr": 1e-5},
            {"params": [p for n, p in model.named_parameters() if "FROD_lambda_l" in n], "lr": 1e-4},
            {"params": [p for n, p in model.named_parameters() if "score" in n], "lr": 1e-5},
        ])
        print("Optimizer for FROD initialized.")
    else:  # Default for LoRA and Full FT
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
        print("Optimizer for LoRA/FullFT initialized.")


    total_training_steps = (len(train_dataloader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps * num_epochs

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_training_steps),
        num_training_steps=total_training_steps
    )
    print(f"LR Scheduler initialized. Total training steps (after accumulation): {total_training_steps}")


    scaler = GradScaler()
    print("Initialized GradScaler for mixed precision training.")

    # 6. Training and Evaluation Loop
    best_accuracy = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        # --- Training ---
        model.train()
        total_loss = 0.0
        processed_batches_in_epoch = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            if not batch or not all(
                    key in batch and batch[key].numel() > 0 for key in ['input_ids', 'attention_mask', 'labels']):
                print(
                    f"Warning: Skipping training batch {batch_idx} due to empty or incomplete data. Batch keys: {batch.keys() if batch else 'Empty Batch'}. Shapes: {[f'{k}:{v.shape}' for k, v in batch.items() if isinstance(v, torch.Tensor)] if batch else 'N/A'}")
                continue

            try:

                batch = {k: v.to(device) for k, v in batch.items()}

                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss = loss / gradient_accumulation_steps


                scaler.scale(loss).backward()

                processed_batches_in_epoch += 1


                if processed_batches_in_epoch % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(
                        train_dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()


                    wandb.log({"train/loss": loss.item() * gradient_accumulation_steps,
                               "train/learning_rate": lr_scheduler.get_last_lr()[0]},
                              step=global_step)
                    global_step += 1

                total_loss += loss.item() * gradient_accumulation_steps
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

            except Exception as e:
                print(f"Error during training batch {batch_idx}: {e}. Skipping this batch.")
                print(
                    f"Batch content leading to error: { {k: v.shape if isinstance(v, torch.Tensor) else len(v) for k, v in batch.items()} }")
                continue


        if processed_batches_in_epoch == 0:
            avg_train_loss = 0.0
        else:
            actual_optimizer_steps = (
                                                 processed_batches_in_epoch + gradient_accumulation_steps - 1) // gradient_accumulation_steps
            if actual_optimizer_steps > 0:
                avg_train_loss = total_loss / actual_optimizer_steps
            else:
                avg_train_loss = 0.0

        print(f"Epoch {epoch + 1} Average Train Loss: {avg_train_loss:.4f}")

        # --- Evaluation ---
        model.eval()
        metric = evaluate.load("accuracy")
        eval_start_time = time.time()
        num_eval_samples = 0
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            if not batch or not all(
                    key in batch and batch[key].numel() > 0 for key in ['input_ids', 'attention_mask', 'labels']):
                print(
                    f"Warning: Skipping evaluation batch {batch_idx} due to empty or incomplete data. Batch keys: {batch.keys() if batch else 'Empty Batch'}. Shapes: {[f'{k}:{v.shape}' for k, v in batch.items() if isinstance(v, torch.Tensor)] if batch else 'N/A'}")
                continue

            try:

                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.no_grad():
                    with autocast():
                        outputs = model(**batch)

                predictions = torch.argmax(outputs.logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
                num_eval_samples += predictions.size(0)
            except Exception as e:
                print(f"Error during evaluation batch {batch_idx}: {e}. Skipping this batch.")
                print(
                    f"Batch content leading to error: { {k: v.shape if isinstance(v, torch.Tensor) else len(v) for k, v in batch.items()} }")
                continue

        eval_time = time.time() - eval_start_time
        eval_metric = metric.compute()
        print(f"Epoch {epoch + 1} Eval Accuracy: {eval_metric['accuracy']:.4f}")

        wandb.log({
            "eval/accuracy": eval_metric['accuracy'],
            "eval/runtime": eval_time,
            "eval/samples_per_second": num_eval_samples / eval_time if eval_time > 0 else 0,
            "epoch": epoch + 1
        }, step=global_step)

        # Save the best model
        if eval_metric['accuracy'] > best_accuracy:
            best_accuracy = eval_metric['accuracy']
            print(f"New best accuracy: {best_accuracy:.4f}. Saving model...")
            # Use save_pretrained for PEFT models
            model.save_pretrained(f"./result/llama7b-{method}-{dataset_name}")
            tokenizer.save_pretrained(f"./result/llama7b-{method}-{dataset_name}")

    print("Training finished.")
    print(f"Best evaluation accuracy: {best_accuracy:.4f}")
    wandb.finish()


# --- Script Entry Point ---
if __name__ == "__main__":
    # piqa_preop, openbookqa, winogrande, siqa, hellaswag, boolq, arc_challenge, arc_easy
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA for sequence classification tasks.")
    parser.add_argument("--method", type=str, default="FROD", choices=["fullft", "lora", "FROD"],
                        help="Fine-tuning method.")
    parser.add_argument("--dataset", type=str, default="siqa", help="Dataset to use for training.")
    parser.add_argument("--model_path", type=str, default="/your/path/to/models/Llama-2-7b-hf",
                        help="Path to the base LLaMA model.")
    # The device argument is kept for consistency but device_map="auto" is the primary mechanism
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Primary device for data.")

    args = parser.parse_args()
    device = torch.device(args.device)

    main(method=args.method, dataset_name=args.dataset, device=device, model_path=args.model_path)