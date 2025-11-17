import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformers import AutoImageProcessor, CLIPForImageClassification, CLIPFeatureExtractor
from peft import get_peft_model, LoraConfig, FRODConfig, VeraConfig
from datasets import load_dataset
from torch.utils.data import DataLoader

import argparse
import sys
from tqdm.auto import tqdm
from torch.optim import AdamW


# --- Auxiliary function: Get PEFT Configuration (including FROD) ---
def get_peft_config(method: str, lora_r: int, lora_alpha: int, lora_dropout: float, lora_target_modules: str,
                    lora_modules_to_save: str, random_seed: int):
    target_modules_list = lora_target_modules.split(',')
    modules_to_save_list = lora_modules_to_save.split(',') if lora_modules_to_save else []

    if method.lower() == "frod":
        return FRODConfig(
            # r=768, # If explicit r is needed
            target_modules=target_modules_list,
            save_projection=False,
            sparse_rate=0.01,  # Sparsity rate, can be passed as argument or fixed
            projection_prng_key=random_seed,
            bias="none",
            modules_to_save=modules_to_save_list,
        )
    elif method.lower() == "vera":
        return VeraConfig(
            r=768,
            save_projection=False,
            target_modules=target_modules_list,
            vera_dropout=lora_dropout,
            bias="none",
            modules_to_save=modules_to_save_list,
        )
    elif method.lower() == "pissa":
        print("Using PissaConfig.")
        return LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            init_lora_weights="pissa_niter_16",
            target_modules=target_modules_list,
            lora_dropout=lora_dropout,
            bias="none",
            modules_to_save=modules_to_save_list,
        )
    elif method.lower() == "lora":
        return LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules_list,
            lora_dropout=lora_dropout,
            bias="none",
            modules_to_save=modules_to_save_list,
        )
    else:
        raise ValueError(f"Unknown PEFT method: {method}. Supported methods: 'frod', 'lora'")


# --- 1. collate_fn remains unchanged ---
def collate_fn(batch):
    pixel_values = torch.stack([x['pixel_values'].squeeze(0) for x in batch])
    labels = torch.tensor([x['labels'] for x in batch])
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }


# --- 2. Custom training function for all models (Full Fine-tune, LoRA, FROD) ---
def custom_train_model(model_name: str, model, train_dataset, args, device):
    print(f"\n--- Training {model_name} Model (Custom Training Loop) ---")

    # Ensure model is on device
    model.to(device)
    model.train()

    # Determine parameters to optimize based on model type (Full FT vs. PEFT)
    # For simplicity, we'll try to get all trainable parameters.
    # For FROD, we explicitly define different learning rates for different param groups.
    # For LoRA/Full FT, one learning rate is usually sufficient.

    if "FROD" in model_name:
        lambda_S_params = [p for n, p in model.named_parameters() if "FROD_lambda_S" in n and p.requires_grad]
        lambda_l_params = [p for n, p in model.named_parameters() if "FROD_lambda_l" in n and p.requires_grad]
        classifier_params = [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad]

        optimizer = AdamW(
            [
                {"params": lambda_S_params, "lr": 1e-4},
                {"params": lambda_l_params, "lr": 5e-4},
                {"params": classifier_params, "lr": 1e-4},
            ]
        )
    elif "VeRA" in model_name:
        optimizer = AdamW(
            [
                {"params": [p for n, p in model.named_parameters() if "vera_lambda_" in n], "lr": 5e-3},
                {"params": [p for n, p in model.named_parameters() if "score" in n], "lr": 1e-4},
            ]
        )
    elif "LoRA" in model_name or "PISSA" in model_name:
        optimizer = AdamW(model.parameters(), lr=1e-4)
    else:  # For Full FT and LoRA
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    train_losses = []
    epochs_log = []  # Renamed to avoid conflict with `epochs` in main function
    min_train_loss = float('inf')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=args.num_workers)

    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"{model_name} Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            # Ensure batch is on device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            # For mixed precision
            if args.use_fp16 and device.type == 'cuda':
                from torch.cuda.amp import autocast, GradScaler
                if not hasattr(model, '_scaler'):  # Initialize scaler once
                    model._scaler = GradScaler()
                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                model._scaler.scale(loss).backward()
                model._scaler.step(optimizer)
                model._scaler.update()
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_epoch_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_epoch_loss)
        epochs_log.append(epoch + 1)
        min_train_loss = min(min_train_loss, avg_epoch_loss)
        print(f"{model_name} Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

    return model, train_losses, epochs_log, min_train_loss


# --- 3. Loss landscape visualization functions ---
def get_random_directions_for_single_model(model):
    directions = []
    for _ in range(2):
        direction = []
        model_device = next((p.device for p in model.parameters() if p.requires_grad), None)
        if model_device is None:
            model_device = torch.device("cpu")

        for param in model.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param, device=model_device)
                norm = torch.norm(noise)
                if norm > 0:
                    noise /= norm
                direction.append(noise)
        directions.append(direction)
    return directions


def get_loss_surface(model, directions, data_batch, x_coords, y_coords, device):
    if len(directions) != 2:
        raise ValueError("The 'directions' argument must contain exactly two direction vectors.")

    loss_surface = np.zeros((len(x_coords), len(y_coords)))

    original_weights = [p.clone().detach().to(device) for p in model.parameters() if p.requires_grad]

    if not original_weights:
        print("Warning: Model has no trainable parameters. Cannot compute loss surface. Returning NaN surface.")
        return np.full((len(x_coords), len(y_coords)), np.nan)

    if len(directions[0]) != len(original_weights) or len(directions[1]) != len(original_weights):
        print(f"Error: Length mismatch between directions and model's trainable parameters.")
        print(f"Direction 0 length: {len(directions[0])}, Direction 1 length: {len(directions[1])}")
        print(f"Model trainable parameters count: {len(original_weights)}")
        # This error is critical, better to raise it
        raise ValueError("Direction vectors must match the number of trainable parameters in the model.")

    batch = {k: v.to(device) for k, v in data_batch.items()}

    model.to(device)  # Ensure model is on device

    for i, x_val in enumerate(x_coords):
        for j, y_val in enumerate(y_coords):
            with torch.no_grad():
                param_idx = 0
                for param in model.parameters():
                    if param.requires_grad:
                        delta = directions[0][param_idx] * x_val + directions[1][param_idx] * y_val
                        # Ensure operations happen on the same device
                        param.data = original_weights[param_idx] + delta.to(original_weights[param_idx].device)
                        param_idx += 1

            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                loss_surface[i, j] = loss.item()

    # Restore original weights
    with torch.no_grad():
        param_idx = 0
        for param in model.parameters():
            if param.requires_grad:
                param.data = original_weights[param_idx]
                param_idx += 1

    return loss_surface


def plot_single_landscape(x, y, z, title, z_min, z_max, fig_num):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)

    surf = ax.plot_surface(X, Y, z.T, cmap='viridis', linewidth=0, antialiased=True, vmin=z_min, vmax=z_max)

    ax.set_xlabel('Direction 1 (α)', fontsize=12)
    ax.set_ylabel('Direction 2 (β)', fontsize=12)
    ax.set_zlabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_zlim(z_min, z_max)

    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, label='Loss Value')
    plt.tight_layout()
    plt.show()


def main(args, test_label):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    try:
        processor = AutoImageProcessor.from_pretrained(args.model_id)
    except Exception as e:
        print(f"Error loading image processor from '{args.model_id}': {e}")
        print("Please check the model_id path or Hugging Face Hub ID.")
        sys.exit(1)

    try:
        dataset = load_dataset(args.dataset_name)
    except Exception as e:
        print(f"Could not load dataset '{args.dataset_name}' from the Hub. Error: {e}")
        print("Please ensure the dataset path or Hugging Face Hub ID is correct and accessible.")
        sys.exit(1)

    if 'train' not in dataset:
        print("Error: 'train' split not found in the dataset.")
        sys.exit(1)

    print("\n--- Dataset Features ---")
    print(dataset['train'].features)
    print("------------------------\n")

    labels = dataset['train'].features['label'].names
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    feature_extractor = CLIPFeatureExtractor.from_pretrained(args.model_id)

    def transform_and_prepare(example):
        if 'image' in example:
            images = [example['image'].convert("RGB")]
        elif 'img' in example:
            images = [example['img'].convert("RGB")]
        else:
            raise KeyError(
                "Neither 'image' nor 'img' found in dataset example. Please check your dataset's image column name.")

        inputs = feature_extractor(images, return_tensors='pt')
        inputs['labels'] = example['label']
        return inputs

    # Using a larger subset for better training results
    prepared_ds_train = dataset["train"].select(range(args.dataset_subset_size)).map(transform_and_prepare,
                                                                                     batched=False)
    # prepared_ds_train = dataset["train"].map(transform_and_prepare,batched=False)
    prepared_ds_train.set_format(type='torch', columns=['pixel_values', 'labels'])

    temp_dataloader = DataLoader(prepared_ds_train, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    print("Fetching a single batch of data from training set for loss landscape calculation...")
    try:
        data_batch_for_landscape = next(iter(temp_dataloader))
    except StopIteration:
        print(
            "Error: No data available in training dataset for loss landscape calculation. Please check your dataset and batch size.")
        sys.exit(1)
    except Exception as e:
        print(f"Error fetching data batch for landscape: {e}")
        sys.exit(1)

    if 'pixel_values' not in data_batch_for_landscape or 'labels' not in data_batch_for_landscape:
        print("Error: Fetched data batch does not contain 'pixel_values' or 'labels'.")
        sys.exit(1)

    # x = np.linspace(-0.1, 0.1, 31)  # Increased range for better visualization
    # y = np.linspace(-0.1, 0.1, 31)
    x = np.linspace(-0.2, 0.2, 21)
    y = np.linspace(-0.2, 0.2, 21)
    # x = np.linspace(-1, 1, 61)
    # y = np.linspace(-1, 1, 61)
    #
    # print("\n--- Initializing and Training VeRA Model ---")
    # initial_base_model_vera = CLIPForImageClassification.from_pretrained(
    #     args.model_id, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    # )
    # vera_config = get_peft_config("vera", args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules,
    #                               args.lora_modules_to_save, args.random_seed)
    # initial_vera_model = get_peft_model(initial_base_model_vera, vera_config).to(device)
    # print(initial_vera_model)
    # initial_vera_model.print_trainable_parameters()
    #
    #
    # print("\n--- Generating Loss Landscapes for Initial VeRA Model ---")
    # print("Generating directions for initial VeRA model...")
    # initial_vera_directions = get_random_directions_for_single_model(initial_vera_model)
    # initial_vera_loss_surface = get_loss_surface(initial_vera_model, initial_vera_directions,
    #                                              data_batch_for_landscape, x, y, device)
    #
    # print("\n--- Training VeRA Model ---")
    # # Note: custom_train_model already handles model.to(device) inside
    # trained_vera_model, vera_train_losses, vera_epochs, vera_min_loss = \
    #     custom_train_model("VeRA", initial_vera_model, prepared_ds_train, args, device)
    #
    # print("Generating directions for trained VeRA model...")
    # vera_directions = get_random_directions_for_single_model(trained_vera_model)
    # print("Calculating loss surface for trained VeRA model...")
    # vera_loss_surface = get_loss_surface(trained_vera_model, vera_directions, data_batch_for_landscape, x, y, device)
    #
    # # --- Initial and Trained Full Fine-tune Model ---
    # initial_full_ft_model = CLIPForImageClassification.from_pretrained(
    #     args.model_id, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    # ).to(device)
    # for param in initial_full_ft_model.parameters():
    #     param.requires_grad = True  # Ensure all parameters are trainable for Full FT
    #
    # print("\n--- Generating Loss Landscapes for Initial Full Fine-tune Model ---")
    # print("Generating directions for initial Full Fine-tune model...")
    # initial_full_ft_directions = get_random_directions_for_single_model(initial_full_ft_model)
    # initial_full_ft_loss_surface = get_loss_surface(initial_full_ft_model, initial_full_ft_directions,
    #                                                 data_batch_for_landscape, x, y, device)
    #
    # full_ft_model = CLIPForImageClassification.from_pretrained(
    #     args.model_id, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    # )  # Not moved to device yet, `custom_train_model` handles it
    # for param in full_ft_model.parameters():
    #     param.requires_grad = True  # Ensure all parameters are trainable for Full FT
    # print(
    #     f"\nFull Fine-tune model trainable params: {sum(p.numel() for p in full_ft_model.parameters() if p.requires_grad):,}")
    # trained_full_ft_model, full_ft_train_losses, full_ft_epochs, full_ft_min_loss = \
    #     custom_train_model("Full Fine-tune", full_ft_model, prepared_ds_train, args, device)
    #
    # print("Generating directions for trained Full Fine-tune model...")
    # full_ft_directions = get_random_directions_for_single_model(trained_full_ft_model)
    # print("Calculating loss surface for trained Full Fine-tune model...")
    # full_ft_loss_surface = get_loss_surface(trained_full_ft_model, full_ft_directions, data_batch_for_landscape, x, y,
    #                                         device)
    #
    # # --- Initial and Trained LoRA Model ---
    # initial_base_model_lora = CLIPForImageClassification.from_pretrained(
    #     args.model_id, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    # )
    # initial_lora_config = get_peft_config("lora", args.lora_r, args.lora_alpha, args.lora_dropout,
    #                                       args.lora_target_modules, args.lora_modules_to_save, args.random_seed)
    # initial_lora_model = get_peft_model(initial_base_model_lora, initial_lora_config).to(device)
    #
    # print("\n--- Generating Loss Landscapes for Initial LoRA Model ---")
    # print("Generating directions for initial LoRA model...")
    # initial_lora_directions = get_random_directions_for_single_model(initial_lora_model)
    # initial_lora_loss_surface = get_loss_surface(initial_lora_model, initial_lora_directions, data_batch_for_landscape,
    #                                              x, y, device)
    #
    # lora_config = get_peft_config("lora", args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules,
    #                               args.lora_modules_to_save, args.random_seed)
    # base_model = CLIPForImageClassification.from_pretrained(
    #     args.model_id, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    # )
    # lora_model = get_peft_model(base_model, lora_config).to(device)
    # lora_model.print_trainable_parameters()
    # trained_lora_model, lora_train_losses, lora_epochs, lora_min_loss = \
    #     custom_train_model("LoRA Fine-tune", lora_model, prepared_ds_train, args, device)
    #
    # print("Generating directions for trained LoRA model...")
    # lora_directions = get_random_directions_for_single_model(trained_lora_model)
    # print("Calculating loss surface for trained LoRA model...")
    # lora_loss_surface = get_loss_surface(trained_lora_model, lora_directions, data_batch_for_landscape, x, y, device)
    #
    # # --- Initial and Trained PISSA Model ---
    # initial_base_model_pissa = CLIPForImageClassification.from_pretrained(
    #     args.model_id, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    # )
    # initial_pissa_config = get_peft_config("pissa", args.lora_r, args.lora_alpha, args.lora_dropout,
    #                                       args.lora_target_modules, args.lora_modules_to_save, args.random_seed)
    # initial_pissa_model = get_peft_model(initial_base_model_pissa, initial_pissa_config).to(device)
    #
    # print("\n--- Generating Loss Landscapes for Initial PISSA Model ---")
    # print("Generating directions for initial PISSA model...")
    # initial_pissa_directions = get_random_directions_for_single_model(initial_pissa_model)
    # initial_pissa_loss_surface = get_loss_surface(initial_pissa_model, initial_pissa_directions, data_batch_for_landscape,
    #                                              x, y, device)
    #
    # pissa_config = get_peft_config("pissa", args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules,
    #                               args.lora_modules_to_save, args.random_seed)
    # base_model = CLIPForImageClassification.from_pretrained(
    #     args.model_id, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    # )
    # pissa_model = get_peft_model(base_model, pissa_config).to(device)
    # pissa_model.print_trainable_parameters()
    # trained_pissa_model, pissa_train_losses, pissa_epochs, pissa_min_loss = \
    #     custom_train_model("PISSA Fine-tune", pissa_model, prepared_ds_train, args, device)
    #
    # print("Generating directions for trained PISSA model...")
    # pissa_directions = get_random_directions_for_single_model(trained_pissa_model)
    # print("Calculating loss surface for trained PISSA model...")
    # pissa_loss_surface = get_loss_surface(trained_pissa_model, pissa_directions, data_batch_for_landscape, x, y, device)

    # --- Initial and Trained FROD Model ---
    print("\n--- Initializing and Training FROD Model ---")
    initial_base_model_frod = CLIPForImageClassification.from_pretrained(
        args.model_id, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )
    frod_config = get_peft_config("frod", args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules,
                                   args.lora_modules_to_save, args.random_seed)
    initial_frod_model = get_peft_model(initial_base_model_frod, frod_config).to(device)
    initial_frod_model.print_trainable_parameters()

    print("\n--- Generating Loss Landscapes for Initial FROD Model ---")
    print("Generating directions for initial FROD model...")
    initial_frod_directions = get_random_directions_for_single_model(initial_frod_model)
    initial_frod_loss_surface = get_loss_surface(initial_frod_model, initial_frod_directions,
                                                  data_batch_for_landscape, x, y, device)

    print("\n--- Training FROD Model ---")
    # Note: custom_train_model already handles model.to(device) inside
    trained_frod_model, frod_train_losses, frod_epochs, frod_min_loss = \
        custom_train_model("FROD", initial_frod_model, prepared_ds_train, args, device)

    print("Generating directions for trained FROD model...")
    frod_directions = get_random_directions_for_single_model(trained_frod_model)
    print("Calculating loss surface for trained FROD model...")
    frod_loss_surface = get_loss_surface(trained_frod_model, frod_directions, data_batch_for_landscape, x, y, device)

    # --- Plot Training Loss Learning Curves ---
    # plt.figure(figsize=(12, 8))
    # plt.plot(full_ft_epochs, full_ft_train_losses, label='Full FT Train Loss', color='blue', linestyle='-')
    # plt.plot(lora_epochs, lora_train_losses, label='LoRA Train Loss', color='red', linestyle='-')
    # plt.plot(pissa_epochs, pissa_train_losses, label='PISSA Train Loss', color='yellow', linestyle='--')
    # plt.plot(vera_epochs, vera_train_losses, label='VeRA Train Loss', color='green', linestyle='--')
    # plt.title('Training Loss Over Epochs', fontsize=16)
    # plt.xlabel('Epoch', fontsize=12)
    # plt.ylabel('Loss', fontsize=12)
    # plt.legend(fontsize=10)
    # plt.grid(True)
    # plt.suptitle('Comparison of Training Dynamics: Full FT vs. LoRA vs. FROD', fontsize=18)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # --- Report Training Loss Lower Bound Comparison ---
    print("\n--- Training Loss Lower Bound Comparison ---")
    # print(f"Full Fine-tune Lowest Training Loss: {full_ft_min_loss:.4f}")
    # print(f"LoRA Fine-tune Lowest Training Loss: {lora_min_loss:.4f}")
    # print(f"VeRA Lowest Training Loss: {vera_min_loss:.4f}")

    # --- Plot Individual Loss Landscapes ---
    print("\n--- Generating Individual Loss Landscapes for Initial and Trained Models ---")
    all_surfaces = {
        # "Initial Full Fine-tune": initial_full_ft_loss_surface,
        # "Trained Full Fine-tune": full_ft_loss_surface,
        # "Initial LoRA": initial_lora_loss_surface,
        # "Trained LoRA": lora_loss_surface,
        # "Initial VERA": initial_vera_loss_surface,
        # "Trained VERA": vera_loss_surface,
        # "Initial PISSA": initial_pissa_loss_surface,
        # "Trained PISSA": pissa_loss_surface,
        "Initial FROD": initial_frod_loss_surface,
        "Trained FROD": frod_loss_surface,
    }
    # Added checks for NaN before min/max to avoid warnings if all values are NaN
    def get_min_max_if_not_nan(arr):
        valid_values = arr[~np.isnan(arr)]
        if valid_values.size > 0:
            return np.min(valid_values), np.max(valid_values)
        return float('nan'), float('nan')

    # min_val, max_val = get_min_max_if_not_nan(initial_full_ft_loss_surface)
    # print(f"Initial Full Fine-tune Landscape Loss: Min = {min_val:.4f}, Max = {max_val:.4f}")
    # plot_single_landscape(x, y, initial_full_ft_loss_surface, "Initial Full Fine-tune Loss Landscape", min_val, max_val, 1)
    #
    # min_val, max_val = get_min_max_if_not_nan(initial_pissa_loss_surface)
    # print(f"Initial PiSSA Fine-tune Landscape Loss: Min = {min_val:.4f}, Max = {max_val:.4f}")
    # plot_single_landscape(x, y, initial_pissa_loss_surface, "Initial PISSA Fine-tune Loss Landscape", min_val, max_val, 2)
    #
    # min_val, max_val = get_min_max_if_not_nan(initial_lora_loss_surface)
    # print(f"Initial LoRA Fine-tune Landscape Loss: Min = {min_val:.4f}, Max = {max_val:.4f}")
    # plot_single_landscape(x, y, initial_lora_loss_surface, "Initial LoRA Fine-tune Loss Landscape", min_val, max_val,
    #                       3)
    #
    # min_val, max_val = get_min_max_if_not_nan(initial_vera_loss_surface)
    # print(f"Initial VeRA Landscape Loss: Min = {min_val:.4f}, Max = {max_val:.4f}")
    # plot_single_landscape(x, y, initial_vera_loss_surface, "Initial VeRA Loss Landscape", min_val, max_val, 4)
    #
    # min_val, max_val = get_min_max_if_not_nan(full_ft_loss_surface)
    # print(f"Trained Full Fine-tune Landscape Loss: Min = {min_val:.4f}, Max = {max_val:.4f}")
    # plot_single_landscape(x, y, full_ft_loss_surface, "Trained Full Fine-tune Loss Landscape",  min_val, max_val, 5)
    #
    # min_val, max_val = get_min_max_if_not_nan(pissa_loss_surface)
    # print(f"Trained PiSSA Fine-tune Landscape Loss: Min = {min_val:.4f}, Max = {max_val:.4f}")
    # plot_single_landscape(x, y, pissa_loss_surface, "Trained PISSA Fine-tune Loss Landscape",  min_val, max_val, 6)
    #
    # min_val, max_val = get_min_max_if_not_nan(lora_loss_surface)
    # print(f"Trained LoRA Fine-tune Landscape Loss: Min = {min_val:.4f}, Max = {max_val:.4f}")
    # plot_single_landscape(x, y, lora_loss_surface, "Trained LoRA Fine-tune Loss Landscape", min_val, max_val, 7)
    #
    # min_val, max_val = get_min_max_if_not_nan(vera_loss_surface)
    # print(f"Trained VeRA Landscape Loss: Min = {min_val:.4f}, Max = {max_val:.4f}")
    # plot_single_landscape(x, y, vera_loss_surface, "Trained VeRA Loss Landscape",  min_val, max_val, 8)
    fig_num = 1
    for title, surface in all_surfaces.items():
        min_val, max_val = get_min_max_if_not_nan(surface)
        print(f"{title} Landscape Loss: Min = {min_val:.4f}, Max = {max_val:.4f}")
        filename = f"{title.replace(' ', '_').lower()}_0.01_surface_{test_label}.npy"
        np.save(filename, surface)
        plot_single_landscape(x, y, surface, f"{title} Loss Landscape", min_val, max_val, fig_num)
        fig_num += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Full Fine-tune, LoRA Fine-tune, and FROD Training Dynamics and Loss Landscapes (Training Loss Only)")
    parser.add_argument("--model_id", type=str, default="/your/path/to/models/clip-vit-base-patch32",
                        help="Path or Hugging Face Hub ID of the pretrained CLIP model.")
    parser.add_argument("--dataset_name", type=str, default="/your/path/to/datasets/stanford_cars",
                        help="Name of the image classification dataset from the Hugging Face Hub.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for the optimizer (for Full FT/LoRA).")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Whether to use mixed precision (fp16) training.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading.")
    parser.add_argument("--dataset_subset_size", type=int, default=1000,
                        help="Size of the subset of the training dataset to use.")  # Added this argument
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension (r).")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.02,
                        help="LoRA dropout probability.")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj, v_proj, o_proj,fc1,fc2",
                        help="Comma-separated list of module names to apply LoRA/FROD to.")
    parser.add_argument("--lora_modules_to_save", type=str, default="classifier",  # Changed from classifier.linear
                        help="Comma-separated list of module names to keep trainable for LoRA/FROD. Use 'classifier' for CLIP.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility, especially for FROD.")

    args = parser.parse_args()
    for i in range(4):
        main(args,i)