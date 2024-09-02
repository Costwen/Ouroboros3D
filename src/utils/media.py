import torch
import numpy as np
from PIL import Image


def get_bg_color(bg_color):
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


def load_image(img_path, img_wh, bg_color, rescale=True, return_type="np"):
    img = np.array(Image.open(img_path).resize(img_wh))
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
