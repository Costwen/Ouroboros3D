import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import imageio
import lightning
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import wandb
from diffusers import (
    StableVideoDiffusionPipeline,
    UNetSpatioTemporalConditionModel,
)
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
)
from einops import rearrange, repeat
from lightning_utilities.core.rank_zero import rank_zero_only
from PIL import Image
from torch import inf, nn
from torch.cuda.amp import autocast
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from safetensors import safe_open
from src.models.ema import LitEma
from src.models.unet.models.unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionModel,
)
from src.utils import RankedLogger
from src.utils.geometry import get_position_map

from ..base import BaseSystem

log = RankedLogger(__name__, rank_zero_only=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
            norm_type,
        )
    return total_norm


def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    clip_grad=True,
) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    device = grads[0].device

    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
            norm_type,
        )

    if clip_grad:
        if error_if_nonfinite and torch.logical_or(
            total_norm.isnan(), total_norm.isinf()
        ):
            raise RuntimeError(
                f"The total norm of order {norm_type} for gradients from "
                "`parameters` is non-finite, so it cannot be clipped. To disable "
                "this error and scale the gradients by the non-finite norm anyway, "
                "set `error_if_nonfinite=False`"
            )
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))

    return total_norm


def _get_add_time_ids(
    unet,
    fps,
    motion_bucket_id,
    noise_aug_strength,
    dtype,
    batch_size,
):
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
    passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids)
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids


def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


class SVDSystem(BaseSystem):
    def __init__(
        self,
        lr: float,
        mv_model: torch.nn.Module,
        recon_model: torch.nn.Module,
        base_model_id: str = "stabilityai/stable-video-diffusion-img2vid",
        variant: str = "fp16",
        recon_model_path: str = "pretrain/LGM/model_fp16.safetensors",
        cfg: float = 0.1,
        report_to: str = "wandb",
        ema_decay_rate: float = 0.9999,
        compile: bool = False,
        use_ema: bool = False,
        # Inference params
        num_inference_steps: int = 25,
        recon_every_n_steps: int = 1,
        vis_inter_results: bool = False,
        cond_types: List[str] = ["rgb", "ccm"],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["mv_model", "recon_model"])

        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            base_model_id, variant=variant
        )
        # self.pipeline.enable_model_cpu_offload()

        self.scheduler = self.pipeline.scheduler
        self.image_encoder = self.pipeline.image_encoder
        self.vae = self.pipeline.vae
        self.feature_extractor = self.pipeline.feature_extractor
        self.image_processor = self.pipeline.image_processor
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.pipeline.unet.requires_grad_(True)
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            base_model_id, subfolder="unet", variant=variant
        )

        self.mv_model = mv_model(unet)
        self.mv_model.set_use_memory_efficient_attention_xformers(True)
        self.mv_model.set_gradient_checkpointing(True)
        self.mv_model.train()
        self.recon_model = recon_model


        tensors = {}

        with safe_open(recon_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        missing_keys, unexpected_keys = self.recon_model.load_state_dict(tensors, strict=False)

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure()
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_ssim = StructuralSimilarityIndexMeasure()
        self.test_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.mv_model, decay=ema_decay_rate)

        self.trainable_parameters = [
            (self.mv_model.parameters(), 1.0),
            (self.recon_model.parameters(), 1.0),
        ]

        self.num_inference_steps = num_inference_steps
        self.recon_every_n_steps = recon_every_n_steps
        self.vis_inter_results = vis_inter_results
        self.cond_types = cond_types
        self.min_guidance_scale = 1.0
        self.max_guidance_scale = 3.0
        self.conditioning_dropout_prob = cfg

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.hparams.compile and stage == "fit":
            self.mv_model = torch.compile(self.mv_model)

        self.log_image = None
        if isinstance(self.logger, lightning.pytorch.loggers.TensorBoardLogger):
            self.log_image = self.tensorboard_log_image
        elif isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.log_image = self.wandb_log_image
            self.logger.watch(self.mv_model, log_graph=False)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers for training."""
        param_groups = []
        for params, lr_scale in self.trainable_parameters:
            param_groups.append({"params": params, "lr": self.hparams.lr * lr_scale})

        optimizer = torch.optim.AdamW(param_groups)
        return optimizer

    def forward(
        self,
        latents,
        timestep,
        encoder_hidden_states,
        added_time_ids,
        cond,
        cameras=None,
    ):
        return self.mv_model(
            latents,
            timestep,
            encoder_hidden_states,
            added_time_ids,
            cond,
            cameras=cameras,
        )

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def training_step(self, batch, batch_idx):
        condition_image = batch["condition_image"]  # cond image b x c x h x w
        diffusion_images = batch["diffusion_images"]  # b x m x c x h x w #  [-1, 1]
        bsz, m, c, h, w = diffusion_images.shape
        dtype = diffusion_images.dtype
        latents = self.tensor_to_vae_latent(diffusion_images, self.vae)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        cond_sigmas = rand_log_normal(shape=[bsz, 1, 1, 1], loc=-3.0, scale=0.5).to(
            latents.device
        )
        noise_aug_strength = cond_sigmas[0]  # TODO: support batch > 1

        condition_image = (
            torch.randn_like(condition_image) * cond_sigmas + condition_image
        )

        conditional_latents = self.tensor_to_vae_latent(condition_image, self.vae)
        conditional_latents = conditional_latents / self.vae.config.scaling_factor

        # Sample a random timestep for each image
        # P_mean=0.7 P_std=1.6
        sigmas = rand_log_normal(shape=[bsz, 1, 1, 1, 1], loc=1.0, scale=1.6).to(
            latents.device
        )
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = latents + noise * sigmas
        timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(
            latents.device
        )

        inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

        # Get the text embedding for conditioning.
        encoder_hidden_states = self._encode_cond(condition_image)

        added_time_ids = _get_add_time_ids(
            self.mv_model.unet,
            7 - 1,  # fixed
            127,  # motion_bucket_id = 127, fixed
            noise_aug_strength,  # noise_aug_strength == cond_sigmas
            encoder_hidden_states.dtype,
            bsz,
        )
        added_time_ids = added_time_ids.to(latents.device)

        # Conditioning dropout to support classifier-free guidance during inference. For more details
        # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
        if self.conditioning_dropout_prob is not None:
            random_p = torch.rand(bsz, device=latents.device)
            # Sample masks for the edit prompts.
            prompt_mask = random_p < 2 * self.conditioning_dropout_prob
            prompt_mask = prompt_mask.reshape(bsz, 1, 1)
            # Final text conditioning.
            null_conditioning = torch.zeros_like(encoder_hidden_states)
            encoder_hidden_states = torch.where(
                prompt_mask, null_conditioning, encoder_hidden_states
            )
            # Sample masks for the original images.
            image_mask_dtype = conditional_latents.dtype
            image_mask = 1 - (
                (random_p >= self.conditioning_dropout_prob).to(image_mask_dtype)
                * (random_p < 3 * self.conditioning_dropout_prob).to(image_mask_dtype)
            )
            image_mask = image_mask.reshape(bsz, 1, 1, 1)
            # Final image conditioning.
            conditional_latents = image_mask * conditional_latents

        # Concatenate the `conditional_latents` with the `noisy_latents`.
        conditional_latents = conditional_latents.unsqueeze(1).repeat(
            1, noisy_latents.shape[1], 1, 1, 1
        )
        inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)

        # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
        target = latents

        c_out = -sigmas / ((sigmas**2 + 1) ** 0.5)
        c_skip = 1 / (sigmas**2 + 1)

        weighing = (1 + sigmas**2) * (sigmas**-2.0)

        model_pred_0 = self.forward(
            inp_noisy_latents.to(dtype),
            timesteps,
            encoder_hidden_states,
            added_time_ids=added_time_ids,
            cond=[
                torch.zeros((bsz, m, c, h, w), device=latents.device, dtype=dtype)
                for _ in range(2)
            ],
            cameras={"intrinsics": batch["intrinsics"], "extrinsics": batch["c2w"]},
        )  # add depth

        pred_x0_0 = model_pred_0 * c_out + c_skip * noisy_latents

        loss = torch.mean(
            (weighing.float() * (pred_x0_0.float() - target.float()) ** 2)
        )

        with torch.no_grad():
            pred_images = self.pipeline.decode_latents(
                pred_x0_0.to(dtype), num_frames=diffusion_images.shape[1]
            )

        pred_images = rearrange(pred_images, "b c m h w -> b m c h w").to(dtype)
        render_x1 = self.recon_model(pred_images, batch, timesteps)
        recon_images = diffusion_images * 0.5 + 0.5
        render_x0 = self.recon_model(diffusion_images, batch, 0)

        recon_loss0 = torch.abs(render_x0["images_pred"] - recon_images).mean()
        recon_loss1 = torch.abs(render_x1["images_pred"] - recon_images).mean()

        with autocast(dtype=torch.float32):
            recon_loss0 += self.lpips(
                render_x0["images_pred"].reshape(-1, 3, h, w).float(),
                recon_images.reshape(-1, 3, h, w).float(),
            )
            recon_loss1 += self.lpips(
                render_x1["images_pred"].reshape(-1, 3, h, w).float(),
                recon_images.reshape(-1, 3, h, w).float(),
            )
        recon_loss = (recon_loss0 + recon_loss1) / 2
        if torch.rand(1) > 0.5:
            # cond = torch.concat([render_x1['images_pred'], render_x1['depths_pred']], axis=2).detach()
            # cond = render_x1['images_pred']
            depth = render_x1["depths_pred"]
            position_map = get_position_map(
                depth=depth,
                cam2world_matrix=batch["c2w"],
                intrinsics=batch["intrinsics"],
                resolution=depth.shape[-1],
            )
            # cond = position_map
            cond = [render_x1["images_pred"].to(dtype), position_map.to(dtype)]
            model_pred_cond = self.forward(
                inp_noisy_latents.to(dtype),
                timesteps,
                encoder_hidden_states,
                added_time_ids=added_time_ids,
                cond=cond,
                cameras={"intrinsics": batch["intrinsics"], "extrinsics": batch["c2w"]},
            )
            # TODO: add render_x1
            pred_x0_1 = model_pred_cond * c_out + c_skip * noisy_latents
            loss = torch.mean(
                (weighing.float() * (pred_x0_1.float() - target.float()) ** 2)
            )
        self.log("train_loss", loss, prog_bar=True)
        self.log("recon_loss", recon_loss, prog_bar=True)
        self.log("recon_loss0", recon_loss0, prog_bar=True)
        self.log("recon_loss1", recon_loss1, prog_bar=True)
        # print("GPU memory:", torch.cuda.memory_allocated() / 1024 ** 3)
        return loss + recon_loss

    def inference_step(self, batch, batch_idx, dataloader_idx=0, stage="val"):
        with self.ema_scope():
            images_pred, cond = self._generate_images(
                batch
            )  # image in [0, 1] 1 x 8 x 3 x 512 x 512
        images_gt = batch["diffusion_images"] * 0.5 + 0.5
        image_fp = self._save_image(
            images_pred,
            images_gt,
            cond,
            ["prompt"],
            f"{dataloader_idx}_{batch_idx}_{self.global_rank}",
            stage=stage,
        )

        images_gt = rearrange(images_gt, "b m c h w -> (b m) c h w")
        images_pred = rearrange(images_pred, "b m c h w -> (b m) c h w")
        cond = rearrange(cond, "b m c h w -> (b m) c h w")

        with autocast(dtype=torch.float32):
            psnr = self.psnr(images_gt.float(), images_pred.float())
            ssim = self.ssim(images_gt.float(), images_pred.float())
            lpips = self.lpips(images_gt.float(), images_pred.float())

            cond_psnr = self.psnr(images_gt.float(), cond.float())
            cond_ssim = self.ssim(images_gt.float(), cond.float())
            cond_lpips = self.lpips(images_gt.float(), cond.float())

        self.log(
            f"{stage}_psnr",
            psnr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_ssim",
            ssim,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_lpips",
            lpips,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=True,
            sync_dist=True,
        )

        self.log(
            f"{stage}_cond_psnr",
            cond_psnr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_cond_ssim",
            cond_ssim,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_cond_lpips",
            cond_lpips,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=True,
            sync_dist=True,
        )
        # save cond
        return image_fp

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.inference_step(batch, batch_idx, dataloader_idx, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        save_flag = (
            f"{batch['id'][0]}"
            if "id" in batch
            else f"{dataloader_idx}_{batch_idx}_{self.global_rank}"
        )
        with self.ema_scope():
            if self.vis_inter_results:
                images_pred, cond, recon_result = self._generate_images_visual(
                    batch, save_flag
                )  # image in [0, 1] 1 x 8 x 3 x 512 x 512
            else:
                images_pred, cond, recon_result = self._generate_images(
                    batch, include_result=True, every_n_steps=self.recon_every_n_steps
                )
        images_gt = batch["diffusion_images"] * 0.5 + 0.5
        image_fp = self._save_image(
            images_pred,
            images_gt,
            cond,
            ["prompt"],
            save_flag,
            stage="test",
        )

        images_gt = rearrange(images_gt, "b m c h w -> (b m) c h w")
        images_pred = rearrange(images_pred, "b m c h w -> (b m) c h w")
        cond = rearrange(cond, "b m c h w -> (b m) c h w")

        with autocast(dtype=torch.float32):
            recon_psnr = self.test_psnr(images_gt.float(), cond.float())
            recon_ssim = self.test_ssim(images_gt.float(), cond.float())
            recon_lpips = self.test_lpips(images_gt.float(), cond.float())

            mv_psnr = self.test_psnr(images_gt.float(), images_pred.float())
            mv_ssim = self.test_ssim(images_gt.float(), images_pred.float())
            mv_lpips = self.test_lpips(images_gt.float(), images_pred.float())

        self.log(
            "test_recon_psnr", recon_psnr, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test_recon_ssim", recon_ssim, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test_recon_lpips", recon_lpips, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test_mv_psnr", mv_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_mv_ssim", mv_ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_mv_lpips", mv_lpips, on_step=False, on_epoch=True, prog_bar=True)

        # save gs
        gaussians = recon_result["gaussians"]

        save_name = f"test_{self.global_step}_{save_flag}"
        self.recon_model.gs.save_ply(
            gaussians, os.path.join(self.save_dir, save_name + ".ply")
        )
        images = []
        elevation = 0
        azimuth = np.arange(0, 360, 2, dtype=np.int32)
        from kiui.cam import orbit_camera

        device = batch["diffusion_images"].device
        fovy: float = 49.1
        tan_half_fov = np.tan(0.5 * np.deg2rad(fovy))
        znear: float = 0.5
        # camera far plane
        zfar: float = 2.5
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
        proj_matrix[0, 0] = 1 / tan_half_fov
        proj_matrix[1, 1] = 1 / tan_half_fov
        proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
        proj_matrix[3, 2] = -(zfar * znear) / (zfar - znear)
        proj_matrix[2, 3] = 1
        cam_radius: float = 1.5

        for azi in tqdm(azimuth):
            cam_poses = (
                torch.from_numpy(
                    orbit_camera(elevation, azi, radius=cam_radius, opengl=True)
                )
                .unsqueeze(0)
                .to(device)
            )

            cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses).transpose(1, 2)  # [V, 4, 4]
            cam_view_proj = cam_view @ proj_matrix  # [V, 4, 4]
            cam_pos = -cam_poses[:, :3, 3]  # [V, 3]

            image = self.recon_model.gs._render(
                gaussians,
                cam_view.unsqueeze(0),
                cam_view_proj.unsqueeze(0),
                cam_pos.unsqueeze(0),
                scale_modifier=1,
            )["image"]
            images.append(
                (
                    image.squeeze(1)
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .float()
                    .cpu()
                    .numpy()
                    * 255
                ).astype(np.uint8)
            )

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(
            os.path.join(self.save_dir, save_name + ".mp4"), images, fps=30
        )

    @torch.no_grad()
    def _encode_cond(self, image, do_classifier_free_guidance=False):
        device, dtype = image.device, image.dtype
        image = image.to(torch.float32)
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0
        # Normalize the image with for CLIP input
        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        # NOTE: set dtype to torch.float16 if needed
        image = image.to(device).to(torch.float16)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)  # b x 1 x 768

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])
        return image_embeddings

    @torch.no_grad()
    def tensor_to_vae_latent(self, t, vae, needs_upcasting=True, micro_bs=1):
        ori_shape_len = len(t.shape)
        model_dtype = next(vae.parameters()).dtype
        if ori_shape_len == 4:
            t = t.unsqueeze(1)
        video_length = t.shape[1]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        if needs_upcasting:
            vae.to(dtype=torch.float32)
            t = t.to(torch.float32)
        # latents = vae.encode(t).latent_dist.sample()
        chunk_outs = []
        t_list = t.chunk(micro_bs, dim=0)
        for t_chunk in t_list:
            chunk_outs.append(vae.encode(t_chunk).latent_dist.sample())
        latents = torch.cat(chunk_outs, dim=0)

        if needs_upcasting:
            vae.to(dtype=model_dtype)
            latents = latents.to(model_dtype)
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
        if ori_shape_len == 4:
            latents = latents.squeeze(1)
        latents = latents * vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def _generate_images(self, batch, include_result=False, every_n_steps=None):
        do_classifier_free_guidance = self.max_guidance_scale > 1.0

        condition_image = batch["condition_image"]  # cond image b x c x h x w
        diffusion_images = batch["diffusion_images"]  # b x m x c x h x w

        (b, m, c, h, w), device = diffusion_images.shape, diffusion_images.device
        # num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        num_frames = m
        dtype = diffusion_images.dtype
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps
        image_embeddings = self._encode_cond(
            condition_image, do_classifier_free_guidance
        )
        # print(image_embeddings)

        # noise = randn_tensor(condition_image.shape, device=condition_image.device)
        condition_image = condition_image + torch.randn_like(condition_image) * 0.02
        cond_image_latent = self.tensor_to_vae_latent(condition_image, self.vae)
        cond_image_latent = cond_image_latent / self.vae.config.scaling_factor

        cond_image_latent = repeat(
            cond_image_latent, "b c h w -> b f c h w", f=num_frames
        )
        if do_classifier_free_guidance:
            cond_image_latent = torch.cat(
                [torch.zeros_like(cond_image_latent), cond_image_latent]
            )

        added_time_ids = _get_add_time_ids(
            self.mv_model.unet,
            7 - 1,
            127,
            0.02,
            dtype,
            b,
        ).to(device)

        latents = (
            torch.randn((b, num_frames, 4, h // 8, w // 8), device=device, dtype=dtype)
            * self.scheduler.init_noise_sigma
        )

        guidance_scale = torch.linspace(
            self.min_guidance_scale, self.max_guidance_scale, num_frames
        ).unsqueeze(0)
        guidance_scale = guidance_scale.to(device)
        guidance_scale = rearrange(guidance_scale, "b m -> b m 1 1 1")

        self._num_timesteps = len(timesteps)
        added_time_ids = (
            torch.cat([added_time_ids] * 2)
            if do_classifier_free_guidance
            else added_time_ids
        )

        # cond = torch.zeros_like(diffusion_images)
        cond = [
            torch.zeros((b, m, c, h, w), device=device, dtype=dtype) for _ in range(2)
        ]

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate image_latents over channels dimention
            latent_model_input = torch.cat(
                [latent_model_input, cond_image_latent], dim=2
            )

            # predict the noise residual
            cond = (
                [torch.cat([cond_] * 2) for cond_ in cond]
                if do_classifier_free_guidance
                else cond
            )

            noise_pred = self.forward(
                latent_model_input.to(dtype),
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                cond=cond,
                cameras=(
                    {"intrinsics": batch["intrinsics"], "extrinsics": batch["c2w"]}
                    if not do_classifier_free_guidance
                    else {
                        "intrinsics": torch.cat([batch["intrinsics"]] * 2),
                        "extrinsics": torch.cat([batch["c2w"]] * 2),
                    }
                ),
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            output = self.scheduler.step(noise_pred, t, latents)
            pred_x0_latent = output.pred_original_sample.to(dtype)
            pred_x0 = self.pipeline.decode_latents(pred_x0_latent, num_frames=m)
            pred_x0 = rearrange(pred_x0, "b c m h w -> b m c h w").to(dtype)

            recon_results = self.recon_model(pred_x0, batch, t)
            # cond = torch.cat([recon_results['images_pred'], recon_results['depths_pred']], axis=2).detach().to(dtype)
            depth = recon_results["depths_pred"]
            position_map = get_position_map(
                depth=depth,
                cam2world_matrix=batch["c2w"],
                intrinsics=batch["intrinsics"],
                resolution=depth.shape[-1],
            )
            # visualize position_map
            # cond = recon_results['images_pred'].detach().to(dtype)
            # cond = position_map.detach().to(dtype)
            cond = [recon_results["images_pred"].to(dtype), position_map.to(dtype)]
            if "rgb" not in self.cond_types:
                cond[0] = torch.zeros_like(cond[0])
            if "ccm" not in self.cond_types:
                cond[1] = torch.zeros_like(cond[1])

            latents = output.prev_sample
            # compute the previous noisy sample x_t -> x_t-1

            if every_n_steps is not None and i % every_n_steps != 0:
                cond = [
                    torch.zeros((b, m, c, h, w), device=device, dtype=dtype)
                    for _ in range(2)
                ]

        cond = recon_results["images_pred"].to(dtype)
        latents = latents.to(dtype)
        frames = self.pipeline.decode_latents(latents, num_frames=m)  # b c m h w
        frames = rearrange(frames, "b c m h w -> (b m) c h w")
        images_pred = self.image_processor.postprocess(frames, output_type="pt")
        images_pred = rearrange(images_pred, "(b m) c h w -> b m c h w", b=b, m=m)

        if not include_result:
            return images_pred, cond[:, :, :3]
        else:
            return images_pred, cond[:, :, :3], recon_results

    @torch.no_grad()
    def _generate_images_visual(self, batch, save_flag):
        do_classifier_free_guidance = self.max_guidance_scale > 1.0

        condition_image = batch["condition_image"]  # cond image b x c x h x w
        diffusion_images = batch["diffusion_images"]  # b x m x c x h x w

        (b, m, c, h, w), device = diffusion_images.shape, diffusion_images.device
        # num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        num_frames = m
        dtype = diffusion_images.dtype
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps
        image_embeddings = self._encode_cond(
            condition_image, do_classifier_free_guidance
        )
        # print(image_embeddings)

        # noise = randn_tensor(condition_image.shape, device=condition_image.device)
        condition_image = condition_image + torch.randn_like(condition_image) * 0.02
        cond_image_latent = self.tensor_to_vae_latent(condition_image, self.vae)
        cond_image_latent = cond_image_latent / self.vae.config.scaling_factor

        cond_image_latent = repeat(
            cond_image_latent, "b c h w -> b f c h w", f=num_frames
        )
        if do_classifier_free_guidance:
            cond_image_latent = torch.cat(
                [torch.zeros_like(cond_image_latent), cond_image_latent]
            )

        added_time_ids = _get_add_time_ids(
            self.mv_model.unet,
            7 - 1,
            127,
            0.02,
            dtype,
            b,
        ).to(device)

        latents = (
            torch.randn((b, num_frames, 4, h // 8, w // 8), device=device, dtype=dtype)
            * self.scheduler.init_noise_sigma
        )

        guidance_scale = torch.linspace(
            self.min_guidance_scale, self.max_guidance_scale, num_frames
        ).unsqueeze(0)
        guidance_scale = guidance_scale.to(device)
        guidance_scale = rearrange(guidance_scale, "b m -> b m 1 1 1")

        self._num_timesteps = len(timesteps)
        added_time_ids = (
            torch.cat([added_time_ids] * 2)
            if do_classifier_free_guidance
            else added_time_ids
        )

        # cond = torch.zeros_like(diffusion_images)
        cond = [
            torch.zeros((b, m, c, h, w), device=device, dtype=dtype) for _ in range(2)
        ]
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate image_latents over channels dimention
            latent_model_input = torch.cat(
                [latent_model_input, cond_image_latent], dim=2
            )

            # predict the noise residual
            cond = (
                [torch.cat([cond_] * 2) for cond_ in cond]
                if do_classifier_free_guidance
                else cond
            )

            noise_pred = self.forward(
                latent_model_input.to(dtype),
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                cond=cond,
                cameras=(
                    {"intrinsics": batch["intrinsics"], "extrinsics": batch["c2w"]}
                    if not do_classifier_free_guidance
                    else {
                        "intrinsics": torch.cat([batch["intrinsics"]] * 2),
                        "extrinsics": torch.cat([batch["c2w"]] * 2),
                    }
                ),
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            output = self.scheduler.step(noise_pred, t, latents)
            pred_x0_latent = output.pred_original_sample.to(dtype)
            pred_x0 = self.pipeline.decode_latents(pred_x0_latent, num_frames=m)
            pred_x0 = rearrange(pred_x0, "b c m h w -> b m c h w").to(dtype)

            recon_results = self.recon_model(pred_x0, batch, t)
            # cond = torch.cat([recon_results['images_pred'], recon_results['depths_pred']], axis=2).detach().to(dtype)
            depth = recon_results["depths_pred"]
            position_map = get_position_map(
                depth=depth,
                cam2world_matrix=batch["c2w"],
                intrinsics=batch["intrinsics"],
                resolution=depth.shape[-1],
            )
            # visualize position_map
            # cond = recon_results['images_pred'].detach().to(dtype)
            # cond = position_map.detach().to(dtype)
            cond = [recon_results["images_pred"].to(dtype), position_map.to(dtype)]
            if "rgb" not in self.cond_types:
                cond[0] = torch.zeros_like(cond[0])
            if "ccm" not in self.cond_types:
                cond[1] = torch.zeros_like(cond[1])

            latents = output.prev_sample

            # save pred multi-view
            latents = latents.to(dtype)
            frames = self.pipeline.decode_latents(latents, num_frames=m)  # b c m h w
            frames = rearrange(frames, "b c m h w -> (b m) c h w")
            multiviews = self.image_processor.postprocess(frames, output_type="pt")
            torchvision.utils.save_image(
                multiviews,
                os.path.join(
                    self.save_dir, f"visual_{save_flag}_step{str(i).zfill(2)}_mv.png"
                ),
                nrow=8,
            )

            # save position_map
            position_maps_vis = rearrange(
                position_map, "b f c h w -> (b f) c h w"
            ).cpu()
            torchvision.utils.save_image(
                position_maps_vis,
                os.path.join(
                    self.save_dir, f"visual_{save_flag}_step{str(i).zfill(2)}_ccm.png"
                ),
                nrow=8,
            )

            # save images_pred (color maps)
            rgb_maps_vis = rearrange(
                recon_results["images_pred"], "b f c h w -> (b f) c h w"
            ).cpu()
            torchvision.utils.save_image(
                rgb_maps_vis,
                os.path.join(
                    self.save_dir, f"visual_{save_flag}_step{str(i).zfill(2)}_rgb.png"
                ),
                nrow=8,
            )

            # save gs
            gaussians = recon_results["gaussians"]

            save_name = f"visual_{save_flag}_step{str(i).zfill(2)}_gs"
            self.recon_model.gs.save_ply(
                gaussians, os.path.join(self.save_dir, save_name + ".ply")
            )
            images = []
            elevation = 0
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            from kiui.cam import orbit_camera

            device = batch["diffusion_images"].device
            fovy: float = 49.1
            tan_half_fov = np.tan(0.5 * np.deg2rad(fovy))
            znear: float = 0.5
            # camera far plane
            zfar: float = 2.5
            proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
            proj_matrix[0, 0] = 1 / tan_half_fov
            proj_matrix[1, 1] = 1 / tan_half_fov
            proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
            proj_matrix[3, 2] = -(zfar * znear) / (zfar - znear)
            proj_matrix[2, 3] = 1
            cam_radius: float = 1.5

            for azi in tqdm(azimuth):
                cam_poses = (
                    torch.from_numpy(
                        orbit_camera(elevation, azi, radius=cam_radius, opengl=True)
                    )
                    .unsqueeze(0)
                    .to(device)
                )

                cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2)  # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix  # [V, 4, 4]
                cam_pos = -cam_poses[:, :3, 3]  # [V, 3]

                image = self.recon_model.gs._render(
                    gaussians,
                    cam_view.unsqueeze(0),
                    cam_view_proj.unsqueeze(0),
                    cam_pos.unsqueeze(0),
                    scale_modifier=1,
                )["image"]
                images.append(
                    (
                        image.squeeze(1)
                        .permute(0, 2, 3, 1)
                        .contiguous()
                        .float()
                        .cpu()
                        .numpy()
                        * 255
                    ).astype(np.uint8)
                )

            images = np.concatenate(images, axis=0)
            imageio.mimwrite(
                os.path.join(self.save_dir, save_name + ".mp4"), images, fps=30
            )

        cond = recon_results["images_pred"].to(dtype)
        latents = latents.to(dtype)
        frames = self.pipeline.decode_latents(latents, num_frames=m)  # b c m h w
        frames = rearrange(frames, "b c m h w -> (b m) c h w")
        images_pred = self.image_processor.postprocess(frames, output_type="pt")
        images_pred = rearrange(images_pred, "(b m) c h w -> b m c h w", b=b, m=m)

        return images_pred, cond[:, :, :3], recon_results

    @torch.no_grad()
    @rank_zero_only
    def _save_image(
        self, images_pred, images, cond, prompt, batch_idx, stage="validation"
    ):
        save_dir = self.save_dir
        if self.log_image is not None:
            _images = rearrange(images, "b m c h w -> 1 c (b h) (m w)")
            _cond = rearrange(cond, "b m c h w -> 1 c (b h) (m w)")
            _images_pred = rearrange(images_pred, "b m c h w ->1 c (b h) (m w)")
            _full_image = torch.concat([_images, _cond, _images_pred], axis=2)
            grid = torchvision.utils.make_grid(_full_image, nrow=2)
            self.log_image(
                tag="{}_images/{}".format(stage, batch_idx),
                image_tensor=grid,
            )
        images = rearrange(images, "b m c h w -> (b h) (m w) c")
        cond = rearrange(cond, "b m c h w -> (b h) (m w) c")
        images_pred = rearrange(images_pred, "b m c h w -> (b h) (m w) c")
        full_image = torch.concat([images, cond, images_pred], axis=0)
        full_image = (full_image * 255).cpu().numpy().astype(np.uint8)
        with open(
            os.path.join(save_dir, f"{stage}_{self.global_step}_{batch_idx}.txt"), "w"
        ) as f:
            f.write("\n".join(prompt))

        im = Image.fromarray(full_image)
        im_fp = os.path.join(
            save_dir,
            f"{stage}_{self.global_step}_{batch_idx}--{prompt[0].replace(' ', '_').replace('/', '_')}.png",
        )
        im.save(im_fp)

        return im_fp

    def tensorboard_log_image(self, tag: str, image_tensor):
        self.logger.experiment.add_image(
            tag,
            image_tensor,
            self.trainer.global_step,
        )

    def wandb_log_image(self, tag: str, image_tensor):
        image_dict = {
            tag: wandb.Image(image_tensor),
        }
        self.logger.experiment.log(
            image_dict,
            step=self.trainer.global_step,
        )