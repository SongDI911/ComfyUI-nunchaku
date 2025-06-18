from typing import Callable

import torch
from comfy.ldm.common_dit import pad_to_patch_size
from diffusers import FluxPipeline
from einops import rearrange
from torch import nn

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.utils import cache_context, create_cache_context
#from nunchaku.lora.flux.compose import compose_lora
from nunchaku.pipeline.pipeline_flux_pulid import PuLIDPipeline
from nunchaku.utils import load_state_dict_in_safetensors

from nunchaku.lora.flux.utils import is_nunchaku_format

import torch
import safetensors
import os
import sys
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
# import comfy.lora_convert

def load_state_dict_in_safetensors(
    path: str | os.PathLike[str],
    device: str | torch.device = "cpu",
    filter_prefix: str = "",
    return_metadata: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, str]]:
    state_dict = {}
    with safetensors.safe_open(path, framework="pt", device=device) as f:
        metadata = f.metadata()
        for k in f.keys():
            if "lora_unet_img_in" in k or "lora_unet_txt_in" in k:
                continue
            if filter_prefix and not k.startswith(filter_prefix):
                continue
            state_dict[k.removeprefix(filter_prefix)] = f.get_tensor(k)
    if return_metadata:
        return state_dict, metadata
    else:
        return state_dict


def to_diffusers(input_lora):
    if isinstance(input_lora, str):
        tensors = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        tensors = {k: v for k, v in input_lora.items()}

    ### convert the FP8 tensors to BF16
    for k, v in tensors.items():
        if v.dtype not in [torch.float64, torch.float32, torch.bfloat16, torch.float16]:
            tensors[k] = v.to(torch.bfloat16)

    new_tensors, alphas = FluxLoraLoaderMixin.lora_state_dict(tensors, return_alphas=True)
    new_tensors = convert_unet_state_dict_to_peft(new_tensors)
    # new_tensors = comfy.lora_convert.convert_lora(tensors)
    # new_tensors = convert_unet_state_dict_to_peft(new_tensors)
    return new_tensors


def compose_lora(
    loras: list[tuple[str | dict[str, torch.Tensor], float]], output_path: str | None = None,
    filter_prefix: str = "",
    del_filter_prefixs: list = [],
) -> dict[str, torch.Tensor]:
    if len(loras) == 1:
        if is_nunchaku_format(loras[0][0]) and (loras[0][1] - 1) < 1e-5:
            if isinstance(loras[0][0], str):
                return load_state_dict_in_safetensors(loras[0][0], device="cpu")
            else:
                return loras[0][0]

    composed = {}
    for lora, strength in loras:
        assert not is_nunchaku_format(lora)
        lora = to_diffusers(lora)
        for k, v in list(lora.items()):
            if v.ndim == 1:
                previous_tensor = composed.get(k, None)
                if previous_tensor is None:
                    if "norm_q" in k or "norm_k" in k or "norm_added_q" in k or "norm_added_k" in k:
                        composed[k] = v
                    else:
                        composed[k] = v * strength
                else:
                    assert not ("norm_q" in k or "norm_k" in k or "norm_added_q" in k or "norm_added_k" in k)
                    composed[k] = previous_tensor + v * strength
            else:
                assert v.ndim == 2
                if ".to_q." in k or ".add_q_proj." in k:  # qkv must all exist
                    if "lora_B" in k:
                        continue

                    q_a = v
                    k_a = lora[k.replace(".to_q.", ".to_k.").replace(".add_q_proj.", ".add_k_proj.")]
                    v_a = lora[k.replace(".to_q.", ".to_v.").replace(".add_q_proj.", ".add_v_proj.")]

                    q_b = lora[k.replace("lora_A", "lora_B")]
                    k_b = lora[
                        k.replace("lora_A", "lora_B")
                        .replace(".to_q.", ".to_k.")
                        .replace(".add_q_proj.", ".add_k_proj.")
                    ]
                    v_b = lora[
                        k.replace("lora_A", "lora_B")
                        .replace(".to_q.", ".to_v.")
                        .replace(".add_q_proj.", ".add_v_proj.")
                    ]

                    assert q_a.shape[0] == k_a.shape[0] == v_a.shape[0]
                    assert q_b.shape[1] == k_b.shape[1] == v_b.shape[1]

                    if torch.isclose(q_a, k_a).all() and torch.isclose(q_a, v_a).all():
                        lora_a = q_a
                        lora_b = torch.cat((q_b, k_b, v_b), dim=0)
                    else:
                        lora_a_group = (q_a, k_a, v_a)
                        new_shape_a = [sum([_.shape[0] for _ in lora_a_group]), q_a.shape[1]]
                        lora_a = torch.zeros(new_shape_a, dtype=q_a.dtype, device=q_a.device)
                        start_dim = 0
                        for tensor in lora_a_group:
                            lora_a[start_dim : start_dim + tensor.shape[0]] = tensor
                            start_dim += tensor.shape[0]

                        lora_b_group = (q_b, k_b, v_b)
                        new_shape_b = [sum([_.shape[0] for _ in lora_b_group]), sum([_.shape[1] for _ in lora_b_group])]
                        lora_b = torch.zeros(new_shape_b, dtype=q_b.dtype, device=q_b.device)
                        start_dims = (0, 0)
                        for tensor in lora_b_group:
                            end_dims = (start_dims[0] + tensor.shape[0], start_dims[1] + tensor.shape[1])
                            lora_b[start_dims[0] : end_dims[0], start_dims[1] : end_dims[1]] = tensor
                            start_dims = end_dims

                    lora_a = lora_a * strength

                    new_k_a = k.replace(".to_q.", ".to_qkv.").replace(".add_q_proj.", ".add_qkv_proj.")
                    new_k_b = new_k_a.replace("lora_A", "lora_B")

                    for kk, vv, dim in ((new_k_a, lora_a, 0), (new_k_b, lora_b, 1)):
                        previous_lora = composed.get(kk, None)
                        composed[kk] = vv if previous_lora is None else torch.cat([previous_lora, vv], dim=dim)

                elif ".to_k." in k or ".to_v." in k or ".add_k_proj." in k or ".add_v_proj." in k:
                    continue
                else:
                    if "lora_A" in k:
                        v = v * strength

                    previous_lora = composed.get(k, None)
                    if previous_lora is None:
                        composed[k] = v
                    else:
                        if "lora_A" in k:
                            if previous_lora.shape[1] != v.shape[1]:  # flux.1-tools LoRA compatibility
                                assert "x_embedder" in k
                                expanded_size = max(previous_lora.shape[1], v.shape[1])
                                if expanded_size > previous_lora.shape[1]:
                                    expanded_previous_lora = torch.zeros(
                                        (previous_lora.shape[0], expanded_size),
                                        device=previous_lora.device,
                                        dtype=previous_lora.dtype,
                                    )
                                    expanded_previous_lora[:, : previous_lora.shape[1]] = previous_lora
                                else:
                                    expanded_previous_lora = previous_lora
                                if expanded_size > v.shape[1]:
                                    expanded_v = torch.zeros(
                                        (v.shape[0], expanded_size), device=v.device, dtype=v.dtype
                                    )
                                    expanded_v[:, : v.shape[1]] = v
                                else:
                                    expanded_v = v
                                composed[k] = torch.cat([expanded_previous_lora, expanded_v], dim=0)
                            else:
                                composed[k] = torch.cat([previous_lora, v], dim=0)
                        else:
                            composed[k] = torch.cat([previous_lora, v], dim=1)

                    composed[k] = (
                        v if previous_lora is None else torch.cat([previous_lora, v], dim=0 if "lora_A" in k else 1)
                    )
    return composed


class ComfyFluxWrapper(nn.Module):
    def __init__(
        self,
        model: NunchakuFluxTransformer2dModel,
        config,
        pulid_pipeline: PuLIDPipeline | None = None,
        customized_forward: Callable = None,
        forward_kwargs: dict | None = {},
    ):
        super(ComfyFluxWrapper, self).__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        self.loras = []

        self.pulid_pipeline = pulid_pipeline
        self.customized_forward = customized_forward
        self.forward_kwargs = {} if forward_kwargs is None else forward_kwargs

        self._prev_timestep = None  # for first-block cache
        self._cache_context = None

    def forward(
        self,
        x,
        timestep,
        context,
        y,
        guidance,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            assert isinstance(timestep, float)
            timestep_float = timestep

        model = self.model
        assert isinstance(model, NunchakuFluxTransformer2dModel)

        bs, c, h, w = x.shape
        patch_size = self.config.get("patch_size", 2)
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size
        img_ids = FluxPipeline._prepare_latent_image_ids(bs, h_len, w_len, x.device, x.dtype)
        txt_ids = torch.zeros((context.shape[1], 3), device=x.device, dtype=x.dtype)

        # load and compose LoRA
        if self.loras != model.comfy_lora_meta_list:
            lora_to_be_composed = []
            for _ in range(max(0, len(model.comfy_lora_meta_list) - len(self.loras))):
                model.comfy_lora_meta_list.pop()
                model.comfy_lora_sd_list.pop()
            for i in range(len(self.loras)):
                meta = self.loras[i]
                if i >= len(model.comfy_lora_meta_list):
                    sd = load_state_dict_in_safetensors(meta[0])
                    model.comfy_lora_meta_list.append(meta)
                    model.comfy_lora_sd_list.append(sd)
                elif model.comfy_lora_meta_list[i] != meta:
                    if meta[0] != model.comfy_lora_meta_list[i][0]:
                        sd = load_state_dict_in_safetensors(meta[0])
                        model.comfy_lora_sd_list[i] = sd
                    model.comfy_lora_meta_list[i] = meta
                lora_to_be_composed.append(({k: v for k, v in model.comfy_lora_sd_list[i].items()}, meta[1]))

            composed_lora = compose_lora(lora_to_be_composed, del_filter_prefixs=["lora_unet_img_in", "lora_unet_txt_in"])

            if len(composed_lora) == 0:
                model.reset_lora()
            else:
                if "x_embedder.lora_A.weight" in composed_lora:
                    new_in_channels = composed_lora["x_embedder.lora_A.weight"].shape[1]
                    current_in_channels = model.x_embedder.in_features
                    if new_in_channels < current_in_channels:
                        model.reset_x_embedder()
                model.update_lora_params(composed_lora)

        controlnet_block_samples = None if control is None else [y.to(x.dtype) for y in control["input"]]
        controlnet_single_block_samples = None if control is None else [y.to(x.dtype) for y in control["output"]]

        if self.pulid_pipeline is not None:
            self.model.transformer_blocks[0].pulid_ca = self.pulid_pipeline.pulid_ca

        if getattr(model, "_is_cached", False):
            # A more robust caching strategy
            cache_invalid = False

            # Check if timestamps have changed or are out of valid range
            if self._prev_timestep is None:
                cache_invalid = True
            elif self._prev_timestep < timestep_float + 1e-5:  # allow a small tolerance to reuse the cache
                cache_invalid = True

            if cache_invalid:
                self._cache_context = create_cache_context()

            # Update the previous timestamp
            self._prev_timestep = timestep_float
            with cache_context(self._cache_context):
                if self.customized_forward is None:
                    out = model(
                        hidden_states=img,
                        encoder_hidden_states=context,
                        pooled_projections=y,
                        timestep=timestep,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance if self.config["guidance_embed"] else None,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                    ).sample
                else:
                    out = self.customized_forward(
                        model,
                        hidden_states=img,
                        encoder_hidden_states=context,
                        pooled_projections=y,
                        timestep=timestep,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance if self.config["guidance_embed"] else None,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        **self.forward_kwargs,
                    ).sample
        else:
            if self.customized_forward is None:
                out = model(
                    hidden_states=img,
                    encoder_hidden_states=context,
                    pooled_projections=y,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                ).sample
            else:
                out = self.customized_forward(
                    model,
                    hidden_states=img,
                    encoder_hidden_states=context,
                    pooled_projections=y,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    **self.forward_kwargs,
                ).sample
        if self.pulid_pipeline is not None:
            self.model.transformer_blocks[0].pulid_ca = None

        out = rearrange(
            out,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_len,
            w=w_len,
            ph=patch_size,
            pw=patch_size,
        )
        out = out[:, :, :h, :w]

        self._prev_timestep = timestep_float
        return out
