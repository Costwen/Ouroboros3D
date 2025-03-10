# Ouroboros3D

Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion

## 🏠 [Project Page](https://costwen.github.io/Ouroboros3D/) | [Paper](https://arxiv.org/abs/2406.03184)

TL;DR. _Ouroboros3D, a unified 3D generation framework, which integrates diffusion-based multi-view image generation and 3D reconstruction into a recursive diffusion process. During the multi-view denoising process, the multi-view diffusion model uses the 3D-aware maps rendered by the reconstruction module at the previous timestep as additional conditions._

## 🔨 Method Overview

![image](https://github.com/user-attachments/assets/0f19fa60-faf3-4444-a7ac-47b410be5fe0)


## Updata
- 2025-02-27: We are excited to announce that our paper has been accepted to CVPR 2025! 🎉
- 2024-09-02: We released the baseline model for `svd & lgm` in [checkpoint](https://huggingface.co/huanngzh/Ouroboros3D-SVD-LGM)
- 2024-08-30: we released the training code, inference code, and model checkpoint of our baseline, Ouroboros3D.

## 🔧 Installation

Just one command to prepare training and test environments:
```Bash
pip install -r requirements.txt
```

## Preparation for training

We render 2 16-frame RGBA orbits at 512 × 512. For each orbit, the cameras are
positioned at a randomly sampled elevation between [-5, 30] degrees. You can rendering any 

```Bash
|
|-- 0a0c7e40a66d4fd090f549599f2f2c9d # object id
| |-- render_0000.webp
| |-- ...
| |-- meta.json # meta info
|-- train.txt
|-- ...
```


After preparing the data, you can to modify the config file `configs/mv/svd_lgm_rgb_ccm_lvis.yaml` to meet your needs.

To train on one node, please use:
```Bash
torchrun --nnodes=1 --nproc_per_node=8 train.py --config configs/mv/svd_lgm_rgb_ccm_lvis.yaml
```
or use 

```Bash
./run.sh train
```

You can modify the `run.sh` to meets your requirements.

During training, you will see the validation results in both `outputs` and logger. You can choose `wandb` or `tensorboard`.

## Inference

You need first download the [checkpoint](https://huggingface.co/huanngzh/Ouroboros3D-SVD-LGM) by `python download_weights.py` to download the model in `checkpoint` dir.

```
python inference.py \
--input testset/3d_arena_a_black_t-shirt_with_the_peace_sign_on_it.png \
--checkpoint checkpoint/Ouroboros3D-SVD-LGM \
--output workspace \
--seed 42 \
--config configs/mv/infer.yaml
```

or use 

```Bash
./run.sh inference
```

The inference results will be stored in the `workspace` folder, which will contain three files: `xxx.ply`, `xxx.png`, and `xxx.mp4`.

- `xxx.ply` is the Gaussian splatting result obtained by `LGM`.
- `xxx.mp4` is the 360-degree rendering generated from `xxx.ply`.
- `xxx.png` is the multiview image generated by `SVD``.

## 🤝 Acknowledgement

We appreciate the open source of the following projects:

[diffusers](https://github.com/huggingface/diffusers) &#8194;
[LGM](https://github.com/3DTopia/LGM) &#8194;
[EpiDiff](https://github.com/huanngzh/EpiDiff)

## 📎 Citation

If you find this repository useful, please consider citing:

```
@article{wen2024ouroboros3d,
  title={Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion},
  author={Wen, Hao and Huang, Zehuan and Wang, Yaohui and Chen, Xinyuan and Qiao, Yu and Sheng, Lu},
  journal={arXiv preprint arXiv:2406.03184},
  year={2024}
}
```
