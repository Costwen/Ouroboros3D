import math
import os
import argparse
import gradio as gr
import hydra
import imageio
import kiui
import numpy as np
import rembg
import torch
import torch.nn.functional as F
from src.data.cameras import Cameras
from diffusers import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
)
from einops import rearrange, repeat
from kiui.cam import orbit_camera
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm
from src.models.unet.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from src.utils.geometry import get_position_map
import torchvision
from kiui.op import recenter

bg_remover = rembg.new_session()

def _encode_cond(image, do_classifier_free_guidance=False):
    device, dtype = image.device, image.dtype
    image = image.to(torch.float32)
    image = _resize_with_antialiasing(image, (224, 224))
    image = (image + 1.0) / 2.0
    # Normalize the image with for CLIP input
    image = pipeline.feature_extractor(
        images=image,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values

    image = image.to(device).to(dtype)
    image_embeddings = pipeline.image_encoder(image).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1)  # b x 1 x 768

    if do_classifier_free_guidance:
        negative_image_embeddings = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])
    return image_embeddings


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


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def get_cameras(c2w, fov):
    fov = fov * np.pi / 180
    w2c = torch.linalg.inv(c2w).to(torch.float)
    width = 512
    fx = torch.tensor(
        [fov2focal(fov=fov_x, pixels=width) for fov_x in fov],
        dtype=torch.float32,
    )
    fy = torch.tensor(
        [fov2focal(fov=fov_y, pixels=width) for fov_y in fov],
        dtype=torch.float32,
    )
    width = torch.tensor([width], dtype=torch.float32).expand(c2w.shape[0])
    height = torch.clone(width)

    cameras = Cameras(
        R=w2c[:, :3, :3],
        T=w2c[:, :3, 3],
        fx=fx,
        fy=fy,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height,
        appearance_id=torch.zeros_like(width),
        normalized_appearance_id=torch.zeros_like(width),
        distortion_params=None,
        camera_type=torch.zeros_like(width),
    )
    return cameras


@torch.no_grad
def generate_images(batch, num_inference_steps, do_classifier_free_guidance=True):
    condition_image = batch["condition_image"]  # cond image b x c x h x w
    # interplate to 512 x 512
    condition_image = F.interpolate(
        condition_image, size=(512, 512), mode="bilinear", align_corners=False
    )
    b, c, h, w = condition_image.shape
    m = 8
    num_frames = m
    dtype = condition_image.dtype

    scheduler.set_timesteps(num_inference_steps, device=device)

    timesteps = scheduler.timesteps

    image_embeddings = _encode_cond(condition_image, do_classifier_free_guidance=True)


    condition_image = condition_image + torch.randn_like(condition_image) * 0.02
    cond_image_latent = pipeline._encode_vae_image(
        condition_image, device, 1, do_classifier_free_guidance=False
    ) 

    cond_image_latent = repeat(cond_image_latent, "b c h w -> b f c h w", f=num_frames)
    if do_classifier_free_guidance:
        cond_image_latent = torch.cat(
            [torch.zeros_like(cond_image_latent), cond_image_latent]
        )

    added_time_ids = _get_add_time_ids(
        mv_model.unet,
        7 - 1,
        127,
        0.02,
        dtype,
        b,
    ).to(device)

    latents = (
        torch.randn((b, num_frames, 4, h // 8, w // 8), device=device, dtype=dtype)
        * scheduler.init_noise_sigma
    )

    guidance_scale = (
        torch.linspace(min_guidance_scale, max_guidance_scale, num_frames)
        .unsqueeze(0)
        .to(device)
    )
    guidance_scale = rearrange(guidance_scale, "b m -> b m 1 1 1")

    added_time_ids = (
        torch.cat([added_time_ids] * 2)
        if do_classifier_free_guidance
        else added_time_ids
    )

    cond = [torch.zeros((b, m, c, h, w), device=device, dtype=dtype) for _ in range(2)]
    
    
    with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
        for i, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

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

            noise_pred = mv_model(
                latent_model_input,
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

            output = scheduler.step(noise_pred, t, latents)
            pred_x0_latent = output.pred_original_sample
            pred_x0 = pipeline.decode_latents(pred_x0_latent, num_frames=m)
            pred_x0 = rearrange(pred_x0, "b c m h w -> b m c h w")

            recon_results = recon_model(pred_x0, batch, t)
            depth = recon_results["depths_pred"]

            position_map = get_position_map(
                depth=depth,
                cam2world_matrix=batch["c2w"],
                intrinsics=batch["intrinsics"],
                resolution=depth.shape[-1],
            )
            cond = [recon_results["images_pred"], position_map]
                
            latents = output.prev_sample

        cond = recon_results["images_pred"].to(dtype)

        frames = pipeline.decode_latents(latents, num_frames=m)  # b c m h w
        frames = rearrange(frames, "b c m h w -> (b m) c h w")
        images_pred = pipeline.image_processor.postprocess(frames, output_type="pt")
        images_pred = rearrange(images_pred, "(b m) c h w -> b m c h w", b=b, m=m)

        return images_pred, cond[:, :, :3], recon_results

def process(input_path, input_num_steps=25, input_seed=42):

    # seed
    kiui.seed_everything(input_seed)

    os.makedirs(workspace, exist_ok=True)
    image_name = os.path.basename(input_path).split(".")[0]
    output_image_path = os.path.join(workspace, image_name+'.png')
    output_video_path = os.path.join(workspace, image_name+'.mp4')
    output_ply_path = os.path.join(workspace, image_name+'.ply')

    input_image = kiui.read_image(input_path, mode='uint8')

    # bg removal
    carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
    mask = carved_image[..., -1] > 0

    # recenter
    image = recenter(carved_image, mask, border_ratio=0.2)
    
    # generate mv
    image = image.astype(np.float32) / 255.0 * 2 - 1

    # rgba to rgb white bg
    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
    

    image = rearrange(torch.from_numpy(image), "h w c -> 1 c h w").to(device)

    dtype = image.dtype
    c2w = torch.from_numpy(np.load("hyper/c2w.npy")).unsqueeze(0).to(device).to(dtype)

    intrinsics = (
        torch.from_numpy(np.load("hyper/intrinsics.npy"))
        .unsqueeze(0)
        .to(device)
        .to(dtype)
    )
    fov = torch.from_numpy(np.load("hyper/fov.npy")).unsqueeze(0).to(device).to(dtype)
    batch = {
        "condition_image": image,  # b x c x h x w
        "c2w": c2w,  # b x f x 4 x 4
        "intrinsics": intrinsics,  # b x f x 4 x 4"
        "fov": fov,  # b x f
        "cameras": [get_cameras(c2w[0], fov[0])],
    }
    images_pred, cond, recon_result = generate_images(
        batch, num_inference_steps=input_num_steps
    )
    # save multi-view images
    mv_images = rearrange(images_pred, "b m c h w -> (b m) c h w")
    # mv_images = images_pred
    torchvision.utils.save_image(mv_images, output_image_path, nrow=4)

    gaussians = recon_result["gaussians"]
    recon_model.gs.save_ply(gaussians, output_ply_path)

    images = []
    elevation = 0

    azimuth = np.arange(0, 360, 2, dtype=np.int32)


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

        image = recon_model.gs._render(
            gaussians,
            cam_view.unsqueeze(0),
            cam_view_proj.unsqueeze(0),
            cam_pos.unsqueeze(0),
            scale_modifier=1,
        )["image"]
        images.append(
            (
                image.squeeze(1).permute(0, 2, 3, 1).contiguous().float().cpu().numpy()
                * 255
            ).astype(np.uint8)
        )

    images = np.concatenate(images, axis=0)
    imageio.mimwrite(output_video_path, images, fps=30)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to the input image")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--workspace", type=str, default='workspace', help='Path to results')
    parser.add_argument("--config", type=str, default='hyper/config.yaml', help="Path to the configuration file") 
    parser.add_argument("--seed", type=int, default=43, help="Seed") 
    
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    workspace = args.workspace

    mv_model_ckpt = "mv_model_unet.pth"
    recon_model_ckpt = "recon_model_unet.pth"

    base_model_id: str = (
        "stabilityai/stable-video-diffusion-img2vid"
    )

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    variant = "fp16"

    pipeline = StableVideoDiffusionPipeline.from_pretrained(base_model_id, variant=variant)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        base_model_id, subfolder="unet", variant=variant
    )

    min_guidance_scale = 1.0
    max_guidance_scale = 3.0

    scheduler = pipeline.scheduler
    mv_model = hydra.utils.instantiate(conf.mv_model)
    mv_model = mv_model(unet)
    recon_model = hydra.utils.instantiate(conf.recon_model)

    # Load the model weights
    # mv_model.load_state_dict(torch.load(mv_model_ckpt, map_location="cpu"))
    # recon_model.load_state_dict(torch.load(recon_model_ckpt, map_location="cpu"))

    mv_model = mv_model.to(device)
    recon_model = recon_model.to(device)
    pipeline = pipeline.to(device)

    process(args.input, args.num_inference_steps, args.seed)

