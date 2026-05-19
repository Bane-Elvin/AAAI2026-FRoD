from __future__ import annotations

import os
import warnings
from collections import defaultdict
from dataclasses import asdict
from enum import Enum
from typing import Optional

import numpy as np
from numpy.linalg import qr
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)

from .._buffer_dict import BufferDict
from ..tuners_utils import _maybe_include_all_linear_layers
from .config import FRODConfig
from .layer import FRODLayer, Linear


class FRODModel(BaseTuner):
    prefix: str = "FROD_lambda"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _init_FROD_V_SparseCOO(self, config: FRODConfig, adapter_name: str) -> None:
        model_dir = getattr(config, "model_dir", ".")
        frod_v_path = os.path.join(model_dir, "FROD_v.pth")
        frod_s_indices_path = os.path.join(model_dir, "FROD_s_indices.pth")
        frod_s_size_path = os.path.join(model_dir, "FROD_s_size.pth")

        weights = defaultdict(dict)
        model_config = self.get_model_config(self.model)
        peft_config = self._prepare_adapter_config(config, model_config)
        peft_config = _maybe_include_all_linear_layers(peft_config, self.model)

        required_categories = set()
        for key, module in self.model.named_modules():
            if not self._check_target_module_exists(peft_config, key):
                continue
            if isinstance(module, nn.Linear):
                parts = key.split(".")
                category = parts[-2] + "." + parts[-1]
                if category == "output.dense" and len(parts) >= 3 and parts[-3] == "attention":
                    category = "attention.output"
                required_categories.add(category)

        os.makedirs(model_dir, exist_ok=True)
        cache_exists = (
            os.path.exists(frod_v_path)
            and os.path.exists(frod_s_indices_path)
            and os.path.exists(frod_s_size_path)
        )
        if cache_exists:
            loaded_v = torch.load(frod_v_path, map_location="cpu")
            loaded_s_indices = torch.load(frod_s_indices_path, map_location="cpu")
            loaded_s_size = torch.load(frod_s_size_path, map_location="cpu")
            loaded_categories = set(loaded_v.keys()) if isinstance(loaded_v, dict) else set()
            if required_categories.issubset(loaded_categories):
                self.FROD_V = loaded_v
                self.FROD_S_indices = loaded_s_indices
                self.FROD_S_size = loaded_s_size
                print(
                    f"Loaded cached FROD projections from {frod_v_path}, {frod_s_indices_path}, and {frod_s_size_path}."
                )
                return
            print("Cached FROD projections are incomplete for the current model; regenerating them.")

        print("FROD: building projection matrices and sparse COO masks.")
        self.FROD_V = {}
        self.FROD_S_indices = {}
        self.FROD_S_size = {}

        for key, module in self.model.named_modules():
            if not self._check_target_module_exists(peft_config, key):
                continue
            if isinstance(module, nn.Linear):
                parts = key.split(".")
                layer_idx = None
                if "layers" in parts:
                    try:
                        pos = parts.index("layers")
                        layer_idx = int(parts[pos + 1])
                    except (ValueError, IndexError):
                        layer_idx = None
                if layer_idx is None:
                    for part in parts:
                        if part.isdigit():
                            layer_idx = int(part)
                            break
                if layer_idx is None:
                    warnings.warn(f"FROD: could not infer layer index from key '{key}', skipping.")
                    continue

                category = parts[-2] + "." + parts[-1]
                if category == "output.dense" and parts[-3] == "attention":
                    category = "attention.output"
                weights[layer_idx][category] = module.weight

        categories = set()
        for layer_dict in weights.values():
            categories.update(layer_dict.keys())

        pi = getattr(config, "regularization_alpha", 1e-3)
        for category in categories:
            matrices = []
            for layer_idx in sorted(weights.keys()):
                weight = weights[layer_idx].get(category)
                if weight is None:
                    continue
                matrices.append(weight.detach().to(torch.float32).cpu().numpy())

            if not matrices:
                continue

            stacked = np.vstack(matrices)
            q_matrix, r_matrix = qr(stacked)
            q_slices = []
            start = 0
            for matrix in matrices:
                rows = matrix.shape[0]
                q_slices.append(q_matrix[start : start + rows, :])
                start += rows

            dim = r_matrix.shape[1]
            t_pi = np.zeros((dim, dim), dtype=r_matrix.dtype)
            for q_slice in q_slices:
                q_term = q_slice.T @ q_slice + pi * np.eye(dim)
                t_pi += np.linalg.inv(q_term)
            t_pi /= len(q_slices)

            _, eigenvectors = np.linalg.eigh(t_pi)
            v_matrix = r_matrix.T @ eigenvectors
            example_weight = next(weight for weight in weights.values() if category in weight)[category]
            v_tensor = torch.from_numpy(v_matrix).to(torch.float32)
            if example_weight.dtype == torch.float32:
                v_tensor = v_tensor.to(torch.float16)
            else:
                v_tensor = v_tensor.to(example_weight.dtype)
            v_tensor = v_tensor.to(device="cpu")

            self.FROD_V[category] = BufferDict({}, persistent=config.save_projection)
            self.FROD_V[category][adapter_name] = v_tensor

            in_dim = v_tensor.shape[0]
            sparsity = config.sparse_rate
            rows, cols = torch.meshgrid(torch.arange(in_dim), torch.arange(in_dim), indexing="ij")
            mask_indices = torch.stack([rows.flatten(), cols.flatten()], dim=1)
            non_diag_indices = mask_indices[mask_indices[:, 0] != mask_indices[:, 1]]
            k = min(int(in_dim * in_dim * sparsity), non_diag_indices.shape[0])
            perm = torch.randperm(non_diag_indices.shape[0])[:k]
            selected_idx = non_diag_indices[perm]
            mask = torch.zeros(in_dim, in_dim)
            mask[selected_idx[:, 0], selected_idx[:, 1]] = 1.0
            indices = torch.nonzero(mask, as_tuple=False).t()
            size = torch.tensor([in_dim, in_dim], dtype=torch.long)

            self.FROD_S_indices[category] = BufferDict({}, persistent=config.save_projection)
            self.FROD_S_indices[category][adapter_name] = indices
            self.FROD_S_size[category] = BufferDict({}, persistent=config.save_projection)
            self.FROD_S_size[category][adapter_name] = size

        torch.save(self.FROD_V, frod_v_path)
        torch.save(self.FROD_S_indices, frod_s_indices_path)
        torch.save(self.FROD_S_size, frod_s_size_path)
        print(f"Saved FROD projection cache to {model_dir}.")

    def _pre_injection_hook(self, model: nn.Module, config: FRODConfig, adapter_name: str) -> None:
        self._init_FROD_V_SparseCOO(config, adapter_name)

    def _check_new_adapter_config(self, config: FRODConfig) -> None:
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

        for existing_config in self.peft_config.values():
            if existing_config is config:
                continue
            if existing_config.projection_prng_key != config.projection_prng_key:
                raise ValueError(
                    f"FRoD projection initialization key must be the same for all adapters. Got {config.projection_prng_key=} but "
                    f"previous config had {existing_config.projection_prng_key}."
                )

        save_project_unique_values = sorted({item.save_projection for item in self.peft_config.values()})
        if len(save_project_unique_values) > 1:
            raise ValueError(
                "FRoD projection weights must be saved for all adapters or none, but got multiple different values: "
                f"{save_project_unique_values}"
            )

    @staticmethod
    def _check_target_module_exists(vera_config, key):
        return check_target_module_exists(vera_config, key)

    def _create_and_replace(
        self,
        vera_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        parts = current_key.split(".")
        category = parts[-2] + "." + parts[-1]
        if category == "output.dense" and parts[-3] == "attention":
            category = "attention.output"

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "vera_dropout": vera_config.vera_dropout,
            "fan_in_fan_out": vera_config.fan_in_fan_out,
            "init_weights": vera_config.init_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            "bias": bias,
        }

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                self.FROD_V[category],
                self.FROD_S_indices[category],
                self.FROD_S_size[category],
                vera_config.vera_dropout,
                vera_config.init_weights,
            )
        else:
            new_module = self._create_new_module(
                vera_config,
                self.FROD_V[category],
                self.FROD_S_indices[category],
                self.FROD_S_size[category],
                adapter_name,
                target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)

        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
                new_module.base_layer.to(child.weight.device)
            else:
                new_module.state = child.state
                new_module.to(child.weight.device)

        meta = torch.device("meta")
        for name, module in new_module.named_modules():
            if "FROD_" in name:
                if isinstance(module, BufferDict):
                    continue
                if not any(p.device == meta for p in module.parameters()):
                    module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if self.prefix not in name:
                param.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            if bias == "all":
                for name, param in model.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True
            elif bias == "vera_only":
                for module in model.modules():
                    if isinstance(module, FRODLayer) and hasattr(module, "bias") and module.bias is not None:
                        module.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(
        vera_config,
        FROD_V,
        FROD_S_indices,
        FROD_S_size,
        adapter_name,
        target,
        **kwargs,
    ):
        bias = kwargs.pop("bias", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = vera_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = vera_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )

        return Linear(
            target,
            FROD_V,
            FROD_S_indices,
            FROD_S_size,
            adapter_name,
            bias=bias,
            **kwargs,
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
            config_dict[key] = config
        return config_dict

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                warnings.warn(
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, FRODLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        key_list = [key for key, _ in self.model.named_modules() if "FROD" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def delete_adapter(self, adapter_name: str):
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if "FROD" not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, FRODLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapter[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ):
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self):
        return self._unload_and_optionally_merge(merge=False)
