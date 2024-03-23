import torch
import numpy as np
from utils import (
    entry, device, two_pi, make_tsr, cos, sin, t_max, 
    smooth_step, step, fract, mix,
    xy_to_uv
)


def sdf_circle(xy_coords, center, radius):
    dists = torch.linalg.norm(xy_coords - center, axis=1)
    sdists = dists - radius
    return sdists


def sdf_segment(xy_coords, a, b):
    dp = xy_coords - a
    db = b - a
    proj = (dp * db).sum(0) / (db * db).sum(0)
    proj = torch.clamp(proj, 0., 1.)
    dist = torch.linalg.norm(dp - db * proj, axis=0)
    return dist


def draw_sdf(xy_coords, canvas_wh, t):
    uv = xy_to_uv(xy_coords, canvas_wh)
    sdists = sdf_circle(uv, make_tsr([0, 0]), 0.2)
    base_color = make_tsr([1, 1, 0])
    colors = base_color * (0.6 + 0.4 * cos(20 * sdists)).unsqueeze(-1)
    return colors


def draw_sdf_iq(xy_coords, canvas_wh, t):
    uv = xy_to_uv(xy_coords, canvas_wh)
    return uv_draw_sdf_iq(uv, t)


def uv_draw_sdf_iq(xy_coords, t):
    # sdists = sdf_circle(xy_coords, make_tsr([0, 0]), 0.5)

    # sdists = sdf_segment(
    #     xy_coords.T,
    #     make_tsr([-0.5, 0]).unsqueeze(-1), make_tsr([0.5, 0]).unsqueeze(-1)
    # ) - 0.5
    sdists = sdf_segment(
        xy_coords.T,
        make_tsr([0, -0.5]).unsqueeze(-1), make_tsr([0., 0.5]).unsqueeze(-1)
    ) - 0.5

    colors = torch.ones(len(xy_coords), 3, device=device)
    colors = torch.where(
        (sdists > 0).unsqueeze(-1),  # broadcasting a bit messy
        make_tsr([0.9, 0.6, 0.3]), make_tsr([0.65, 0.85, 1.0])
    )

    colors *= (
        1.0 - torch.exp(-20.0 * abs(sdists))
    ).unsqueeze(-1)

    # colors *= (
    #     0.7 + 0.3 * cos(
    #         (40.0 + 20*cos(2. * t * two_pi)) * sdists
    #     )
    # ).unsqueeze(-1)

    # colors *= (
    #     0.5 + 0.5 * cos(
    #         (40.0 + 20*cos(2. * t * two_pi)) * sdists
    #     )
    # ).unsqueeze(-1)

    colors *= (periodic_bump(
        (10.0 + 5.*cos(2. * t * two_pi)) * sdists,
        thresh=0.3
    )).unsqueeze(-1)

    return colors


def periodic_bump(xs, thresh=0.6):
    symmetric_hat = torch.abs(fract(xs) * 2. - 1.)
    smooth_bump = smooth_step(thresh, 1.0, symmetric_hat)
    return smooth_bump


def draw_sdf_ruler(xy_coords, canvas_wh, t):
    uv = xy_to_uv(xy_coords, canvas_wh)
    # return uv_draw_sdf_ruler(uv, t)
    return uv_sdf_flat_ruler(uv, t)


def uv_draw_sdf_ruler(xy_coords, t):
    xs, ys = xy_coords.T
    thetas = torch.atan2(ys, xs)
    thetas = thetas % two_pi  # converting -ve angle to +ve
    # add time ticking effect here with t

    inc = np.deg2rad(2.0)  # 180 increments
    ticks = torch.round(thetas / inc).type(torch.int)

    ticks_angle = ticks * inc
    ticks_dir_xy = torch.stack([cos(ticks_angle), sin(ticks_angle)], axis=0)
    del ticks_angle

    start_at = 0.95 * torch.ones(len(xy_coords), device=device)
    start_at = torch.where(
        torch.remainder(ticks, 5) == 0,
        0.90, start_at
    )
    start_at = torch.where(
        torch.remainder(ticks, 10) == 0,
        0.85, start_at
    )

    # this scaling routine is not good.
    scaling = 0.55 + 0.45 * cos(t * two_pi)

    sdists = 100. * torch.ones(len(xy_coords), device=device)
    for i in [1, 2, 3]:
        bound = (i * 1/3) * scaling
        _sdists = sdf_segment(
            xy_coords.T, ticks_dir_xy * bound * start_at, ticks_dir_xy * bound
        ) - (bound * 0.006)
        sdists = torch.minimum(sdists, _sdists)

    # sdists = torch.clamp(sdists * 300., 0.0, 1.0)  # *100 for anti-alisting

    # I like this 2nd strategy better;
    # but overall, -ve sdist set to 0. +ve sdist has a gradual change to 1.
    sdists = smooth_step(0, 0.001, sdists)
    # Or do abrupt change without AA
    # sdists = 1.0 - step(0., sdists)

    # colors = make_tsr([0.9, 0.6, 0.3]) * sdists.unsqueeze(-1)
    colors = mix(
        make_tsr([0.65, 0.85, 1.0]), make_tsr([0.9, 0.6, 0.3]), sdists.unsqueeze(-1)
    )
    return colors


def uv_sdf_flat_ruler(xy_coords, t):
    # scaling = 1.0 + 0.5 * cos(t * two_pi)
    scaling = torch.exp(0.8 * sin(2.0 * t * two_pi))

    xs, ys = xy_coords.T
    ones = torch.ones(len(xy_coords), device=device)

    inc = 0.05 / 2
    segment_d = 0.005 / 2
    teeth = 0.15 * scaling
    inc, segment_d = scaling * inc, scaling * segment_d

    # teeth start ratio
    t_s, t_m, t_l = 0.35, 0.65, 1.0

    ticks = torch.round(xs / inc).type(torch.int)
    ticks_x = ticks * inc

    end = 0.9
    y_end = end * ones

    y_start = (end - t_s*teeth) * ones
    y_start = torch.where(torch.remainder(ticks, 5) == 0, end - t_m*teeth, y_start)
    y_start = torch.where(torch.remainder(ticks, 10) == 0, end - t_l*teeth, y_start)

    ticks_start = torch.stack([ticks_x, y_start], axis=0)
    ticks_end = torch.stack([ticks_x, y_end], axis=0)

    sdists_horizontal = sdf_segment(
        xy_coords.T, ticks_start, ticks_end
    ) - segment_d

    ticks = torch.round(ys / inc).type(torch.int)
    ticks_y = ticks * inc

    end = 0.8
    x_end = end * ones

    x_start = (end - t_s*teeth) * ones
    x_start = torch.where(torch.remainder(ticks, 5) == 0, end - t_m*teeth, x_start)
    x_start = torch.where(torch.remainder(ticks, 10) == 0, end - t_l*teeth, x_start)

    ticks_start = torch.stack([x_start, ticks_y], axis=0)
    ticks_end = torch.stack([x_end, ticks_y], axis=0)

    sdists_vertical = sdf_segment(
        xy_coords.T, ticks_start, ticks_end
    ) - segment_d

    sdists = torch.minimum(sdists_horizontal, sdists_vertical)
    sdists = smooth_step(0, 0.005, sdists)

    colors = mix(
        make_tsr([0.65, 0.85, 1.0]), make_tsr([0.9, 0.6, 0.3]), sdists.unsqueeze(-1)
    )

    return colors


def main():
    canvas_wh = (512 * 4, 256 * 4)
    # entry(draw_sdf, canvas_wh, fname="out/04_sdf_circle")
    # return

    # entry(draw_sdf_iq, canvas_wh, fname="out/04_sdf_circle_iq", num_frames=90, fps=30)
    # return

    canvas_wh = (512 * 4, 512 * 2)
    # entry(draw_sdf_ruler, canvas_wh, fname="out/04_sdf_ruler")
    entry(draw_sdf_ruler, canvas_wh, fname="out/04_sdf_ruler", num_frames=180, fps=60)


if __name__ == '__main__':
    main()
