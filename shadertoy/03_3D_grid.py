import torch
import numpy as np
from utils import (
    entry, device, two_pi, make_tsr, to_t,
    cos, sin, t_max, smooth_step, step, fract, coords_take
)

from m_02_2D_grid import integrated_period_step

from hc3d.render import camera_pose, compute_intrinsics


def unproject(K, pixel_coords, depth=1.0):
    """sometimes also referred to as backproject
        pixel_coords: [n, 2] pixel locations
        depth: [n,] or [,] depth value. of a shape that is broadcastable with pix coords
    """
    K = K[0:3, 0:3]

    pixel_coords = as_homogeneous(pixel_coords)
    pixel_coords = pixel_coords.T  # [2+1, n], so that mat mult is on the left

    # this will give points with z = -1, which is exactly what you want since
    # your camera is facing the -ve z axis
    pts = torch.linalg.inv(K) @ pixel_coords

    pts = pts * depth  # [3, n] * [n,] broadcast
    pts = pts.T
    pts = as_homogeneous(pts)
    return pts


def as_homogeneous(pts):
    # pts: [..., d]
    *front, d = pts.shape
    points = torch.ones((*front, d + 1), dtype=pts.dtype, layout=pts.layout, device=pts.device)
    points[..., :d] = pts
    return points


def rays_from_coords(xy_coords, K, c2w_pose, normalize_dir=True):
    ro = c2w_pose[:, -1]
    pts = unproject(K, xy_coords, depth=1)
    pts = pts @ c2w_pose.T
    rd = pts - ro  # equivalently can subtract [0,0,0,1] before pose transform
    rd = rd[:, :3]
    if normalize_dir:
        rd = rd / torch.linalg.norm(rd, axis=-1, keepdims=True)
    ro = torch.tile(ro[:3], (len(rd), 1))
    return ro, rd


def scaling_at_t(t):
    # this scaling makes a lot of sense
    return 2 * torch.exp(1. * sin(2. * t * two_pi))


def checker_alias(xy_coords, t):
    k = scaling_at_t(t)
    xy_coords = xy_coords * k
    xs, ys = period_step(xy_coords, 0.5)
    colors = apply_smooth_logic(xs, ys, "xor")
    rgbs = make_tsr([1.0, 1.0, 1.0]) * colors.unsqueeze(-1)
    return rgbs


def grid_alias(xy_coords, t):
    k = scaling_at_t(t)
    xy_coords = xy_coords * k
    xs, ys = period_step(xy_coords, 0.1)
    colors = apply_smooth_logic(xs, ys, "and")
    rgbs = make_tsr([1.0, 1.0, 1.0]) * colors.unsqueeze(-1)
    return rgbs


def checker_antialias(xy_coords, dx_dy, t):
    k = scaling_at_t(t)
    xy_coords = xy_coords * k

    dx_dy *= k
    xs, ys = integrated_period_step(xy_coords, dx_dy, 0.5)
    colors = apply_smooth_logic(xs, ys, "xor")
    rgbs = make_tsr([1.0, 1.0, 1.0]) * colors.unsqueeze(-1)
    return rgbs


def grid_antialias(xy_coords, dx_dy, t):
    k = scaling_at_t(t)
    xy_coords = xy_coords * k

    dx_dy *= k
    xs, ys = integrated_period_step(xy_coords, dx_dy, 0.1)
    colors = apply_smooth_logic(xs, ys, "and")
    rgbs = make_tsr([1.0, 1.0, 1.0]) * colors.unsqueeze(-1)
    return rgbs


def apply_smooth_logic(xs, ys, logic):
    """the logic is implemented using integrable ops, on top of which filtering is possible"""
    if logic == "and":
        return xs * ys
    elif logic == "or":
        return xs + ys - xs*ys
    elif logic == "xor":
        return xs + ys - 2*xs*ys
    else:
        raise NotImplementedError("unimpl")


def period_step(xy_coords, p):
    xy = step(p, fract(xy_coords))
    xs, ys = xy.T
    return xs, ys


def ray_ground_intersect(ro, rd):
    ro_y = ro[:, 1]
    rd_y = rd[:, 1]
    t = (0.0 - ro_y) / rd_y
    # t = torch.clamp(t, 10000.)

    is_hit = t > 0.
    t = torch.where(is_hit, t, torch.tensor(0.))
    xyz_loc = (ro + t.unsqueeze(-1) * rd)
    return is_hit, t, xyz_loc


def ground_plane(xy_coords, canvas_wh, system_t):
    W, H = canvas_wh.tolist()
    FoV = 53
    lookat = np.array([0, 0, 0])

    # eye = np.array([4., 1., 0])
    angle = 0.5 * np.sin(system_t.item() * two_pi)
    eye = np.array([4. * np.cos(angle), 1., np.sin(angle)])

    up = np.array([0, 1, 0])

    K = compute_intrinsics(W/H, FoV, H)
    pose = camera_pose(eye, lookat - eye, up)
    K, pose = to_t(K, pose)

    ro, rd = rays_from_coords(xy_coords, K, pose)
    ddx_ro, ddx_rd = rays_from_coords(xy_coords + make_tsr([1., 0.]), K, pose)
    ddy_ro, ddy_rd = rays_from_coords(xy_coords + make_tsr([0., 1.]), K, pose)

    is_hit, t, loc = ray_ground_intersect(ro, rd)

    _, _, ddx_loc = ray_ground_intersect(ddx_ro, ddx_rd)
    _, _, ddy_loc = ray_ground_intersect(ddy_ro, ddy_rd)

    uv = coords_take(loc[is_hit], "xz")
    ddx_uv = coords_take(ddx_loc[is_hit], "xz") - uv
    ddy_uv = coords_take(ddy_loc[is_hit], "xz") - uv
    w = t_max(
        torch.stack([torch.abs(ddx_uv), torch.abs(ddy_uv)], dim=0), axis=0
    )

    rgbs = checker_alias(uv, system_t)
    # rgbs = grid_alias(uv, system_t)
    # rgbs = checker_antialias(uv, w, system_t)
    # rgbs = grid_antialias(uv, w, system_t)

    canvas = torch.ones(len(xy_coords), 3, device=device)
    canvas[is_hit] = rgbs

    # add volume fog
    alphas = 1. - torch.exp(-0.025 * t).unsqueeze(-1)
    canvas = alphas * 1. + (1 - alphas) * canvas

    return canvas


def main():
    # canvas_wh = (512 * 3, 256 * 3)
    canvas_wh = (512, 256)
    entry(ground_plane, canvas_wh, num_frames=270, fps=30, fname="out/03_ground")
    pass


if __name__ == "__main__":
    main()
