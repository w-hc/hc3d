import numpy as np
import cv2


def batch_img_resize(imgs, new_h=50):
    # often need to resize for view plane cuz o3d can't handle large images
    import torch
    from torchvision.transforms import Resize
    import einops

    _, h, w, _ = imgs.shape
    ratio = w / h
    new_w = int(new_h * ratio)
    ops = Resize((new_h, new_w), antialias=True)
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


def paint_text_on_img(image, text, xywh=[0.05, 0.05, 0.1, 0.1]):
    '''
    this op modifies in-place
    width of xywh is not enforced; long text runs longer
    '''
    H, W, _ = image.shape
    x, y, w, h = xywh
    x, w = round(x * W), round(w * W)
    y, h = round(y * H), round(h * H)

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)  # red text

    # these factors are tuned manually
    font_scale = h * 0.04
    thickness = round(h * 0.1)

    # Get the size of the text box
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Put the text onto the image, putText requires the bottom left coords
    _img = cv2.putText(image, text, (x, y + text_height), font, font_scale, color, thickness)
    return _img


def inlier_mask(pts, std_multiple=5.0):
    pts = pts[:, :3]
    centroid = pts.mean(axis=0)
    dist = np.linalg.norm(pts - centroid, axis=1)
    mask = (dist < (dist.mean() + dist.std() * std_multiple))

    n_before = len(pts)
    n_after = mask.sum()
    percent = n_after / n_before * 100
    print(f"inlier: retain {n_after} / {n_before}; {percent:.2f}% of points")

    return mask
