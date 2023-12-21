from math import pi
import torch
from torch import sin, cos
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm

device = torch.device("cuda")

two_pi = 2 * pi  # frequently used constant


def entry(
    shading_func,
    canvas_wh=(256, 128), num_frames=1, fps=1,
    fname="out", batch_size=4096,
):
    W, H = canvas_wh
    n = W * H
    ys, xs = np.meshgrid(range(H), range(W), indexing="ij")
    xy_coords = np.stack([xs, ys], axis=-1).reshape(n, 2)

    canvas_wh = torch.tensor([W, H], dtype=int).to(device)
    xy_coords = torch.as_tensor(xy_coords).to(device)

    frames = []
    for i in tqdm(range(num_frames)):
        rgbs = torch.zeros(n, 3, device=device, dtype=torch.float)
        for (s, e) in per_batch_start_end(n, batch_size):
            _rgbs = shading_func(xy_coords[s:e], canvas_wh, i / num_frames)  # NOTE t is fractional in [0, 1]
            _rgbs.clamp_(0, 1)
            rgbs[s:e] = _rgbs
        rgbs = rgbs.reshape(H, W, 3)
        rgbs = (rgbs * 255.).type(torch.uint8).cpu().numpy()
        frames.append(rgbs)

    if len(frames) == 1:
        iio.imwrite(f"{fname}.png", rgbs)
    else:
        # use imageio to save as video
        frames = np.stack(frames, axis=0)
        iio.imwrite(f"{fname}.mp4", frames, fps=fps)


def per_batch_start_end(n, bs):
    # return the tuples of [start, end] indices over each batch
    # taking into account uneven division
    for i in range(int(np.ceil(n / bs))):
        s = i * bs
        e = min(n, s + bs)
        yield (s, e)


def make_tsr(arr):
    return torch.tensor(arr, device=device)