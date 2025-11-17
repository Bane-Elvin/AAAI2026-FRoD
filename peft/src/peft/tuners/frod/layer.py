# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import List, Optional
import numpy as np
from numpy.linalg import qr, inv, eigh, norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class FRODLayer(BaseTunerLayer):
    adapter_layer_names = ("FROD_lambda_S", "FROD_lambda_l")
    other_param_names = ("FROD_V", "FROD_U", "FROD_S_mask")
    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.vera_dropout = nn.ModuleDict({})

        # For storing vector scale
        self.FROD_lambda_S = nn.ParameterDict({})  # adapter_name -> Parameter[in,out]
        self.FROD_lambda_l = nn.ParameterDict({})

        # Stores a reference to the vera_A/B BufferDict.
        # Set to `None` otherwise to avoid computation with random weights
        self.FROD_S_mask: Optional[BufferDict] = None
        self.FROD_V: Optional[BufferDict] = None
        self.FROD_U = nn.ParameterDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name,
        FROD_V: BufferDict,
        FROD_S_mask: BufferDict,
        vera_dropout,
        init_weights,
    ):
        weight = self.get_base_layer().weight
        device = weight.device
        dtype = weight.dtype
        self.r[adapter_name] = self.out_features
        if vera_dropout > 0.0:
            vera_dropout_layer = nn.Dropout(p=vera_dropout)
        else:
            vera_dropout_layer = nn.Identity()

        self.vera_dropout.update(nn.ModuleDict({adapter_name: vera_dropout_layer}))
        # Actual trainable parameters
        in_dim = self.in_features

        self.FROD_lambda_S[adapter_name] = nn.Parameter(torch.randn(in_dim, in_dim))

        # non trainable references to FROD_V buffers
        self.FROD_V = FROD_V
        self.FROD_S_mask = FROD_S_mask
        U, L= self.calculate_FROD_U_FROD_lambda(self.FROD_V[adapter_name], weight)
        U = U
        L = L
        W = torch.zeros_like(weight)
        self.FROD_lambda_l[adapter_name] = nn.Parameter(L, requires_grad = True)
        if init_weights:
            self.reset_vera_parameters(adapter_name)
        self.FROD_U[adapter_name] = nn.Parameter(U, requires_grad = False)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)
        weight = transpose(W.to(dtype), self.fan_in_fan_out)
        self.get_base_layer().weight.data = weight

    def calculate_FROD_U_FROD_lambda(self, V, W):
        w = W.detach().cpu().numpy()
        v = V.detach().cpu().numpy()
        v_inv_T = inv(v).T
        Bi = w @ v_inv_T
        l = np.linalg.norm(Bi, axis=0)
        u = np.divide(Bi, l, where=l > 1e-8)
        # # print(u.T @ u)
        # # print(l)
        # w = w - u @ np.diag(l) @ v.T
        # W = torch.from_numpy(w).float()
        U = torch.from_numpy(u).float()
        L = torch.from_numpy(l).float()
        return U, L

    def reset_vera_parameters(self, adapter_name):
        if adapter_name in self.FROD_lambda_S:
            with torch.no_grad():
                nn.init.zeros_(self.FROD_lambda_S[adapter_name])


class Linear(nn.Linear, FRODLayer):
    # Vera implemented in a dense layer
    def __init__(
        self,
        base_layer,
        FROD_V: BufferDict,
        FROD_S_mask: BufferDict,
        adapter_name: str,
        vera_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        FRODLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, FROD_V, FROD_S_mask, vera_dropout, init_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.vera_lambda_d.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.vera_lambda_d.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        V = self.FROD_V[adapter]
        U = self.FROD_U[adapter]
        mask = self.FROD_S_mask[adapter].to(U.device)
        raw  = self.FROD_lambda_S[adapter].to(U.device)
        S = raw * mask
        l = self.FROD_lambda_l[adapter].to(U.device)
        L = torch.diag_embed(l)

        return transpose(U @ (S + L) @ V.T, self.fan_in_fan_out)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.FROD_lambda_S:
                    continue

                V = self.FROD_V[active_adapter].to(x.device)
                U = self.FROD_U[active_adapter].to(x.device)
                mask = self.FROD_S_mask[active_adapter].to(x.device)
                raw  = self.FROD_lambda_S[active_adapter].to(x.device)
                S = raw * mask
                l = self.FROD_lambda_l[active_adapter].to(x.device)
                L =torch.diag_embed(l)

                x = x.to(V.dtype)
                h = self.vera_dropout[active_adapter](x)
                # result = result + F.linear(F.linear(F.linear(h, V), S), U) + F.linear(l * F.linear(h, V), U)
                # result = F.linear(F.linear(F.linear(h, V.T), L + S), U)
                delta_weight = transpose(U @ (S + L) @ V.T, self.fan_in_fan_out)
                result = result + F.linear(h, delta_weight)


        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "vera." + rep
