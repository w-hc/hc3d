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


def inlier_mask(pts, std_multiple=5.0):
    pts = pts[:, :3]
    centroid = pts.mean(axis=0)
    dist = np.linalg.norm(pts - centroid, axis=1)
    mask = (dist < (dist.mean() + dist.std() * std_multiple))

    n_before = len(pts)
    n_after = mask.sum()
    percent = n_after / n_before * 100
    print(f"inlier: retain {n_after} / {n_after}; {percent:.2f}% of points")

    return mask
