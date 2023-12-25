from math import pi
import torch
from torch import sin, cos, max
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm

device = torch.device("cuda")

two_pi = 2 * pi  # frequently used constant


def entry(
    shading_func,
    canvas_wh=(256, 128), num_frames=1, fps=1,
    fname="out", batch_size=4096 * 100,
):
    W, H = canvas_wh
    n = W * H
    ys, xs = np.meshgrid(range(H), range(W), indexing="ij")
    xy_coords = np.stack([xs, ys], axis=-1).reshape(n, 2)

    canvas_wh = torch.tensor([W, H], dtype=int).to(device)
    xy_coords = torch.as_tensor(xy_coords, dtype=torch.float32).to(device)

    frames = []
    for i in tqdm(range(num_frames)):
        rgbs = torch.zeros(n, 3, device=device, dtype=torch.float)
        for (s, e) in per_batch_start_end(n, batch_size):
            t = torch.tensor(i / num_frames, device=device)  # NOTE t is fractional in [0, 1]
            _rgbs = shading_func(xy_coords[s:e], canvas_wh, t)
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


def t_max(tsr, axis):
    # syntax sugar
    return torch.max(tsr, axis=axis)[0]


def make_tsr(arr):
    # syntax sugar
    return torch.tensor(arr, device=device, dtype=torch.float)


def to_t(*args):
    ret = []
    for elem in args:
        target_dtype = torch.float32 if np.issubdtype(elem.dtype, np.floating) else None
        ret.append(
            torch.as_tensor(elem, dtype=target_dtype, device=device)
        )
    return ret


def smooth_step(edge0, edge1, xs):
    """
    https://thebookofshaders.com/glossary/?search=smoothstep
    Smooth Hermite interpolation between [0, 1] when edge0 < x < edge1
    A threshold function with smooth transition
    """
    ts = (xs - edge0) / (edge1 - edge0)
    ts = torch.clamp(ts, 0., 1.)
    ts = ts * ts * (3. - 2. * ts)
    return ts


def step(edge, xs):
    xs = (xs > edge).type(torch.float32)
    return xs


def fract(xs):
    return xs - torch.floor(xs)


def coords_take(arr, symbols):
    # e.g. "xzxzy" will return tsr of shape [n, 5]
    symb_to_inx = {
        "x": 0, "y": 1, "z": 2
    }
    chosen = [symb_to_inx[s] for s in symbols]
    return arr[:, chosen]
