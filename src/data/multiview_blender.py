import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json
import random

import cv2
from einops import rearrange, repeat
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from .cameras import Cameras
from src.utils.point import fov2focal


def collect_fn(batch):
    cameras = [item.pop("cameras") for item in batch]
    batch_processed = default_collate(batch)
    batch_processed["cameras"] = cameras
    return batch_processed


class MultiViewDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        bg_color: str,
        num_frames: int,
        meta_file: str = "meta.json",
        ids_file: Optional[str] = None,
        img_wh: Tuple[int, int] = (512, 512),
        relative_pose: bool = True,
        num_samples: int = -1,
        repeat: int = 1,
        view_start_idx: Union[int, str] = "random",
    ):
        scenes = []

        with open(ids_file, "r") as f:
            for line in f:
                scene = line.strip()
                scenes.append(os.path.join(root_dir, scene))


        if num_samples is not None and num_samples > 0:
            scenes = scenes[:num_samples]

        self.root_dir = root_dir
        self.scenes = scenes
        self.meta_file = meta_file
        self.img_wh = img_wh
        self.relative_pose = relative_pose
        self.num_frames = num_frames
        self.bg_color = self.get_bg_color(bg_color)
        self.repeat = repeat
        self.view_start_idx = view_start_idx

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            bg_color = np.random.rand(3)
        elif isinstance(bg_color, float):
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(self, img_path, rescale=True, return_type="np"):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        # split root_dir and path
        im = Image.open(img_path)
        bg_color = self.bg_color
        img = np.array(im.resize(self.img_wh))
        img = img.astype(np.float32) / 255.0  # [0, 1]

        if img.shape[-1] == 4:
            alpha = img[..., 3:4]
            img = img[..., :3] * alpha + bg_color * (1 - alpha)
        if rescale:
            img = img * 2.0 - 1.0  # to -1 ~ 1
        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def load_normal(self, normal_path, return_type="np"):
        img = np.array(Image.open(normal_path).resize(self.img_wh))
        img = img.astype(np.float32) / 255.0  # [0, 1]

        if img.shape[-1] == 4:
            alpha = img[..., 3:4]
            img = img[..., :3] * alpha

        img = img * 2.0 - 1.0  # to -1 ~ 1

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError
        return img

    def load_depth(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_NEAREST)
        depth = img[None, :, :, 3]
        mask = depth == 0  # depth = 65535 is invalid
        mask = ~mask
        return depth, mask.astype(np.float32)

    def __len__(self):
        return len(self.scenes) * self.repeat

    def get_cameras(self, c2w, fov_x_list, fov_y_list):
        w2c = torch.linalg.inv(c2w).to(torch.float)
        width = self.img_wh[0]
        fx = torch.tensor(
            [fov2focal(fov=fov_x, pixels=width) for fov_x in fov_x_list],
            dtype=torch.float32,
        )
        fy = torch.tensor(
            [fov2focal(fov=fov_y, pixels=width) for fov_y in fov_y_list],
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

    def get_item(self, index):
        index = index % len(self.scenes)

        scene = self.scenes[index]

        meta = json.load(open(os.path.join(scene, self.meta_file)))
        fov = meta["camera_angle_x"]

        locations = meta["locations"]
        view_start_idx = self.view_start_idx
        if isinstance(view_start_idx, str) and view_start_idx == "random":
            # roll the views
            start_idx = random.randint(0, len(locations) - 1)
        elif isinstance(view_start_idx, int):
            start_idx = view_start_idx
        else:
            raise NotImplementedError
        locations = locations[start_idx:] + locations[:start_idx]

        c2w_list = np.array([item["transform_matrix"] for item in locations])
        select_indices = np.linspace(
            0, len(c2w_list) - 1, self.num_frames, dtype=np.int32
        )
        c2w_list = c2w_list[select_indices]
        focal_length = 0.5 * 1 / np.tan(0.5 * fov)
        intrinsics = np.array(
            [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]],
            dtype=np.float32,
        )
        intrinsics = torch.from_numpy(intrinsics)
        c2w_matrixs = torch.from_numpy(c2w_list)  # N x 4 x 4
        # to opencv
        c2w_matrixs[:, :3, 1:3] *= -1

        num_views = c2w_matrixs.shape[0]
        intrinsics = repeat(intrinsics, "h w -> n h w", n=num_views)

        if self.relative_pose:
            src_w2c = torch.inverse(c2w_matrixs[:1])  # (1, 4, 4)
            src_distance = c2w_matrixs[:1, :3, 3].norm(dim=-1)  # (1)
            canonical_c2w = torch.matmul(src_w2c, c2w_matrixs)  # (Nv, 4, 4) z as x axis
            # shift to origin depth
            shift = torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, src_distance],
                    [0, 0, 0, 1],
                ]
            )
            canonical_c2w = torch.matmul(shift, canonical_c2w)
            c2w_matrixs = canonical_c2w
        fov_x_list = [fov] * num_views
        fov_y_list = [fov] * num_views
        cameras = self.get_cameras(c2w_matrixs, fov_x_list, fov_y_list)
        fov = fov * 180 / np.pi
        fov = torch.tensor([fov] * c2w_matrixs.shape[0], dtype=torch.float32)

        # Load all images
        images = []
        for i, loc in enumerate(locations):
            if i not in select_indices:
                continue
            img_path = os.path.join(scene, loc["frames"][0]["name"])
            img = self.load_image(img_path, return_type="pt").permute(2, 0, 1)
            images.append(img)
        condition_image = images[0]
        diffused_images = torch.stack(images, dim=0)

        return {
            "id": scene.split("/")[-1].rsplit(".", 1)[0],
            "condition_image": condition_image,
            "intrinsics": intrinsics,
            "cameras": cameras,
            "c2w": c2w_matrixs,
            "diffusion_images": diffused_images,
            "fov": fov,
        }

    def __getitem__(self, index):
        index = index % len(self.scenes)

        while True:
            try:
                return self.get_item(index)
            except Exception as e:
                print("Load data error, retrying...")
                index = random.randint(0, len(self.scenes) - 1)


class MultiViewDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset[Any] = None,
        val_dataset: Dataset[Any] = None,
        test_dataset: Dataset[Any] = None,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        real_dataset=None,
        real_batch_size=-1,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train = train_dataset
        self.data_val = val_dataset
        self.data_test = test_dataset
        self.data_real = real_dataset

        self.num_workers = num_workers if num_workers else train_batch_size * 2

    def prepare_data(self) -> None:
        # TODO: check if data is available
        pass

    def _dataloader(
        self, dataset: Dataset, batch_size: int, shuffle: bool
    ) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collect_fn,
        )

    def real_dataloader(self):
        return DataLoader(
            self.data_real,
            batch_size=self.hparams.real_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collect_fn,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.data_train is None:
            return None
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collect_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if isinstance(self.data_val, ListConfig):
            return [
                self._dataloader(dataset, self.hparams.val_batch_size, False)
                for dataset in self.data_val
            ]
        elif isinstance(self.data_val, DictConfig):
            return [
                self._dataloader(dataset, self.hparams.val_batch_size, False)
                for _, dataset in self.data_val.items()
            ]
        else:
            return self._dataloader(self.data_val, self.hparams.val_batch_size, False)

    def test_dataloader(self) -> DataLoader[Any]:
        if isinstance(self.data_test, ListConfig):
            return [
                self._dataloader(dataset, self.hparams.test_batch_size, False)
                for dataset in self.data_test
            ]
        elif isinstance(self.data_test, DictConfig):
            return [
                self._dataloader(dataset, self.hparams.test_batch_size, False)
                for _, dataset in self.data_test.items()
            ]
        else:
            return self._dataloader(self.data_test, self.hparams.test_batch_size, False)


if __name__ == "__main__":

    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    seed_everything(42)
    dataset = MultiViewDataset(
        root_dir="data/lvis",
        meta_file="meta.json",
        ids_file="data/lvis/render-lvis-nv16-ele0.txt",
        bg_color="white",
        num_frames=8,
        img_wh=(512, 512),
    )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collect_fn
    )
    for batch in tqdm(dataloader):
        images = batch["diffusion_images"]
        images = rearrange(images, "b n c h w -> b c h (n w)")
        save_image(images, "diffusion_images_2.png")
        save_image(batch["condition_image"], "condition_image_2.png")
