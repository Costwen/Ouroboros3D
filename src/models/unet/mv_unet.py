from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
# from diffusers import UNetSpatioTemporalConditionModel
from .models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from einops import rearrange, repeat
from omegaconf import ListConfig

from src.utils.project import HarmonicEmbedding, encode_plucker


def ray_sample(intrinsics, extrinsics, resolution):
    """Compute the position map from the depth map and the camera parameters.

    Args:
        depth (torch.Tensor): The depth map with the shape (B, F, 1, H, W).
        intrinsics (torch.Tensor): The camera intrinsics matrix in opencv system.
        extrinsics (torch.Tensor): The camera extrinsics matrix.
        image_wh (Tuple[int, int]): The image width and height.

    Returns:
        torch.Tensor: The position map with the shape (H, W, 3).
    """
    b, f, _, _ = intrinsics.shape
    uv = torch.stack(
        torch.meshgrid(
            torch.arange(resolution, dtype=torch.float32, device=intrinsics.device),
            torch.arange(resolution, dtype=torch.float32, device=intrinsics.device),
            indexing="ij",
        )
    )
    uv = uv * (1.0 * 1 / resolution) + (0.5 * 1 / resolution)
    cam_locs_world = extrinsics[..., :3, 3]  # b x f x 3
    uv = repeat(uv, "c h w -> b f c h w", b=b, f=f)
    x_cam = uv[:, :, 0]
    y_cam = uv[:, :, 1]  # b x f x h x w
    # Compute the position map by back-projecting depth pixels to 3D space
    fx = intrinsics[..., 0, 0].unsqueeze(-1)  # b x f
    fy = intrinsics[..., 1, 1].unsqueeze(-1)  # b x f
    cx = intrinsics[..., 0, 2].unsqueeze(-1)  # b x f
    cy = intrinsics[..., 1, 2].unsqueeze(-1)  # b x f
    z_cam = torch.ones(
        (b, f, resolution, resolution), device=intrinsics.device, dtype=intrinsics.dtype
    )
    # x = (u_coord - intrinsics[..., 0, 2]) * depth / intrinsics[..., 0, 0] # x = (u - cx) * z / fx
    # y = (v_coord - intrinsics[..., 1, 2]) * depth / intrinsics[..., 1, 1] # y = (v - cy) * z / fy

    x_lift = (x_cam - cx.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
    y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

    camera_coords = torch.stack(
        [x_lift, y_lift, z_cam, torch.ones_like(z_cam)], dim=-1
    ).to(
        intrinsics.dtype
    )  # b x f x h x w x 4

    world_coords = torch.einsum(
        "b f i j, b f h w i -> b f h w j", extrinsics, camera_coords
    )[
        ..., :3
    ]  # b x f x h x w x 4
    # world_coords = coords_homogeneous @ extrinsics.T

    cam_locs_world = repeat(
        cam_locs_world, "b f c -> b f h w c", h=resolution, w=resolution
    )
    ray_dirs = world_coords - cam_locs_world
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
    ray_origins = cam_locs_world
    return ray_origins, ray_dirs


class PluckerEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x):
        return self.embed(x)


class MVModel(nn.Module):
    def __init__(
        self,
        unet: UNetSpatioTemporalConditionModel = None,
        cond_encoder: nn.Module = None,
        add_plucker: bool = False,
    ):
        super().__init__()
        self.unet = unet
        # self.unet.disable_gradient_checkpointing()
        self.cond_encoder = (
            nn.ModuleList(cond_encoder)
            if cond_encoder is not None and isinstance(cond_encoder, ListConfig)
            else cond_encoder
        )
        self.add_plucker = add_plucker

        if add_plucker:
            self.harm_embed = HarmonicEmbedding()
            in_ch = self.harm_embed.get_output_dim(6)
            # get all the plucker embeddings
            self.down_plucker = nn.ModuleList()
            ch_list = [320, 320, 640, 1280]
            for out_ch in ch_list:
                self.down_plucker.append(PluckerEmbedding(in_ch, out_ch))
            self.up_plucker = nn.ModuleList()
            ch_list = [1280, 1280, 1280, 640]
            for out_ch in ch_list:
                self.up_plucker.append(PluckerEmbedding(in_ch, out_ch))

            self.mid_plucker = PluckerEmbedding(in_ch, 1280)

    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op=None
    ) -> None:
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def set_gradient_checkpointing(self, valid: bool) -> None:
        def fn_recursive_set_checkpointing(module: torch.nn.Module):
            if hasattr(module, "gradient_checkpointing"):
                if valid:
                    module.gradient_checkpointing = True
                else:
                    module.gradient_checkpointing = False

            for child in module.children():
                fn_recursive_set_checkpointing(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_checkpointing(module)

    def forward(
        self,
        sample: torch.FloatTensor,
        timesteps: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        cond: Optional[Dict[str, Any]] = None,
        cameras: Optional[Dict[str, Any]] = None,
    ):
        if cond is not None and self.cond_encoder is not None:
            if isinstance(cond, list) and isinstance(self.cond_encoder, nn.ModuleList):
                # multiple t2i-adapters
                features = ()
                for cond_, encoder in zip(cond, self.cond_encoder):
                    features += (encoder(cond_, sample.dtype),)

                # Only use single condition
                # cond_feature_ = self.cond_encoder[0](cond[0], sample.dtype)
                # features += (cond_feature_, [0] * len(self.unet.down_blocks))
            else:
                # one t2i-adapter
                features = self.cond_encoder(cond, sample.dtype)
                # reverse the features
                # features = features[::-1]
        else:
            features = [0] * len(self.unet.down_blocks)

        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.to(sample.dtype)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)
        t_emb = self.unet.time_proj(timesteps).to(sample.dtype)
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        emb = self.unet.time_embedding(t_emb)

        time_embeds = self.unet.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.unet.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(
            num_frames, dim=0
        )

        # 2. pre-process
        sample = self.unet.conv_in(sample)

        image_only_indicator = torch.zeros(
            batch_size, num_frames, dtype=sample.dtype, device=sample.device
        )

        down_block_res_samples = (sample,)
        idx = 0
        for i, downsample_block in enumerate(self.unet.down_blocks):
            if self.add_plucker:
                ray_origins, ray_dirs = ray_sample(
                    cameras["intrinsics"], cameras["extrinsics"], sample.shape[-1]
                )
                plucker = encode_plucker(ray_origins, ray_dirs, self.harm_embed)
                plucker = self.down_plucker[i](plucker)
                plucker = rearrange(plucker, "b f h w c-> (b f) c h w")
                sample += plucker

            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                # NEW: multiple t2i-adapters or one t2i-adapter
                if isinstance(features, tuple):
                    adapter_features_ = ()
                    for feat in features:
                        adapter_features_ += (feat[idx],)
                else:
                    adapter_features_ = features[idx]
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    adapter_feature=adapter_features_,
                )
                idx += 1
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )
            down_block_res_samples += res_samples
        # 4. mid
        if self.add_plucker:
            ray_origins, ray_dirs = ray_sample(
                cameras["intrinsics"], cameras["extrinsics"], sample.shape[-1]
            )
            plucker = encode_plucker(ray_origins, ray_dirs, self.harm_embed)
            plucker = self.mid_plucker(plucker)
            plucker = rearrange(plucker, "b f h w c-> (b f) c h w")
            sample += plucker
        sample = self.unet.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # NEW: multiple t2i-adapters or one t2i-adapter
        if isinstance(features, tuple):
            for feat in features:
                sample += feat[-1]
        else:
            sample += features[-1]

        # 5. up
        for i, upsample_block in enumerate(self.unet.up_blocks):
            if self.add_plucker:
                ray_origins, ray_dirs = ray_sample(
                    cameras["intrinsics"], cameras["extrinsics"], sample.shape[-1]
                )
                plucker = encode_plucker(ray_origins, ray_dirs, self.harm_embed)
                plucker = self.up_plucker[i](plucker)
                plucker = rearrange(plucker, "b f h w c-> (b f) c h w")
                sample += plucker
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )
        # 6. post-process
        sample = self.unet.conv_norm_out(sample)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        return sample
