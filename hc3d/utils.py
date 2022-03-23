import numpy as np


def batch_img_resize(imgs, new_h=50):
    # often need to resize for view plane cuz o3d can't handle large images
    import torch
    from torchvision.transforms import Resize
    import einops

    _, h, w, _ = imgs.shape
    ratio = w / h
    new_w = int(new_h * ratio)
    ops = Resize((new_h, new_w))
    imgs = torch.from_numpy(imgs)
    imgs = einops.rearrange(imgs, "b h w c -> b c h w")
    imgs = ops(imgs)
    imgs = einops.rearrange(imgs, "b c h w -> b h w c").numpy()

    if imgs.dtype == np.float32:
        imgs = (imgs * 255).astype(np.uint8)
    elif imgs.dtype == np.uint8:
        pass
    else:
        raise Exception("invalid type")
    return imgs


def outlier_mask(pts, thresh=0.1):
    pts = pts[:, :3]
    mask = (np.abs(pts) > thresh).any(axis=1)
    mask = ~mask
    return mask
