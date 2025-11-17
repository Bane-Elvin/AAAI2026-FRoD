from __future__ import annotations
import os
import numpy as np
from numpy.linalg import qr, inv, eigh, norm
import math
import warnings
from dataclasses import asdict
from enum import Enum
from typing import Optional, Union
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)

from .._buffer_dict import BufferDict
from ..tuners_utils import _maybe_include_all_linear_layers
from .config import FRODConfig
from .layer import Linear, FRODLayer


class FRODModel(BaseTuner):

    prefix: str = "FROD_lambda"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _init_FROD_V_FROD_S_mask(self, config: FRODConfig, adapter_name: str) -> None:

        model_dir = getattr(config, "model_dir", ".")
        frod_v_path = os.path.join(model_dir, "FROD_v.pth")
        frod_s_mask_path = os.path.join(model_dir, "FROD_S_mask.pth")

        if os.path.exists(frod_v_path) and os.path.exists(frod_s_mask_path):
            self.FROD_V = torch.load(frod_v_path)
            self.FROD_S_mask = torch.load(frod_s_mask_path)
            print(f"Already load {frod_v_path} and {frod_s_mask_path}")
            return

        self.FROD_S_mask = dict()
        self.FROD_V = dict()

        weights = defaultdict(dict)
        model_config = self.get_model_config(self.model)

        peft_config = self._prepare_adapter_config(config, model_config)
        peft_config = _maybe_include_all_linear_layers(peft_config, self.model)
        print(self.model)
        for key, module in self.model.named_modules():
            if not self._check_target_module_exists(peft_config, key):
                continue
            print(key)
            if isinstance(module, nn.Linear):
                parts = key.split('.')
                layer_idx = int(parts[3])
                category = parts[-2] + "." + parts[-1]
                if category == "output.dense":
                    if parts[-3] == "attention":
                        category = "attention.output"
                weights[layer_idx][category] = module.weight

        categories = set()
        for layer_dict in weights.values():
            categories.update(layer_dict.keys())

        pi = getattr(config, 'regularization_alpha', 1e-3)

        for category in categories:
            A_list = []
            for layer_idx in sorted(weights.keys()):
                W1 = weights[layer_idx].get(category)
                if W1 is None:
                    continue
                A_list.append(W1.detach().cpu().numpy())

            if not A_list:
                continue

            # QR
            A_stack = np.vstack(A_list)
            Q, R = qr(A_stack)
            Qi_list = []
            m = 0
            for A_i in A_list:
                mi = A_i.shape[0]
                Qi_list.append(Q[m: m + mi, :])
                m += mi

            dim = R.shape[1]
            T_pi = np.zeros((dim, dim), dtype=R.dtype)
            for Qi in Qi_list:
                Qi_term = Qi.T @ Qi + pi * np.eye(dim)
                T_pi += np.linalg.inv(Qi_term)
            T_pi /= len(Qi_list)

            tau, Z = np.linalg.eigh(T_pi)
            V = R.T @ Z
            example_W = next(w for w in weights.values() if category in w)[category]
            V_tensor = torch.from_numpy(V).to(example_W.device).type(example_W.dtype)
            self.FROD_V[category] = BufferDict({}, persistent=config.save_projection)
            self.FROD_V[category][adapter_name] = V_tensor

            # S_mask
            in_dim = V_tensor.shape[0]
            sparsity = config.sparse_rate
            rows, cols = torch.meshgrid(torch.arange(in_dim), torch.arange(in_dim), indexing='ij')
            mask_indices = torch.stack([rows.flatten(), cols.flatten()], dim=1)
            non_diag_indices = mask_indices[mask_indices[:, 0] != mask_indices[:, 1]]
            k = min(int(in_dim * in_dim * sparsity), non_diag_indices.shape[0])
            perm = torch.randperm(non_diag_indices.shape[0])[:k]
            selected_idx = non_diag_indices[perm]
            mask = torch.zeros(in_dim, in_dim)
            mask[selected_idx[:, 0], selected_idx[:, 1]] = 1.0
            self.FROD_S_mask[category] = BufferDict({}, persistent=config.save_projection)
            self.FROD_S_mask[category][adapter_name] = mask


        torch.save(self.FROD_V, frod_v_path)
        torch.save(self.FROD_S_mask, frod_s_mask_path)
        print(f"Already save FROD_V and FROD_S_mask to {frod_v_path}, {frod_s_mask_path}")


    def _pre_injection_hook(self, model: nn.Module, config: FRODConfig, adapter_name: str) -> None:
        self._init_FROD_V_FROD_S_mask(config, adapter_name)


    def _check_new_adapter_config(self, config: FRODConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # the below todo is copied from LoRA
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

        for existing_config in self.peft_config.values():
            if existing_config is config:
                # skip the current config
                continue

            if existing_config.projection_prng_key != config.projection_prng_key:
                raise ValueError(
                    f"Vera PRNG initialisation key must be the same for all adapters. Got {config.projection_prng_key=} but "
                    f"previous config had {existing_config.projection_prng_key}."
                )

        save_project_unique_values = sorted({config.save_projection for config in self.peft_config.values()})
        if len(save_project_unique_values) > 1:
            raise ValueError(
                "VeRA projection weights must be saved for all adapters or none, but got multiple different values: "
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

        parts = current_key.split('.')
        category = parts[-2] + "." + parts[-1]
        if category == "output.dense":
            if parts[-3] == "attention":
                category = "attention.output"

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "vera_dropout": vera_config.vera_dropout,
            "fan_in_fan_out": vera_config.fan_in_fan_out,
            "init_weights": vera_config.init_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        kwargs["bias"] = bias

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                self.FROD_V[category],
                self.FROD_S_mask[category],
                vera_config.vera_dropout,
                vera_config.init_weights,
            )
        else:
            new_module = self._create_new_module(vera_config, self.FROD_V[category], self.FROD_S_mask[category], adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "FROD_" in name:
                if not any(p.device == meta for p in module.parameters()):
                    module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "vera_only":
                for m in model.modules():
                    if isinstance(m, FRODLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(vera_config, FROD_V, FROD_S_mask, adapter_name, target, **kwargs):
        # avoid eager bnb import

        bias = kwargs.pop("bias", False)
        loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.get("loaded_in_4bit", False)

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
        new_module = Linear(
            target,
            FROD_V,
            FROD_S_mask,
            adapter_name,
            bias=bias,
            **kwargs,
        )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

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
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
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
        # we cannot use self.prefix as we want to include non-trainable vera parameters
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
                # save any additional trainable modules part of `modules_to_save`
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def delete_adapter(self, adapter_name: str):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        # we cannot use self.prefix as we want to include non-trainable vera parameters
        key_list = [key for key, _ in self.model.named_modules() if "vera" not in key]
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
        r"""
        This method merges the Vera layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self):
        """
        Gets back the base model by removing all the Vera modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
