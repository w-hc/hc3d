import torch
from utils import entry, device, two_pi, make_tsr, cos, sin, t_max, smooth_step


def grid(xy_coords, canvas_wh, t):
    period_n_pix = 40 + 20 * sin(1.0 * t * two_pi)  # grid size itself changes over time

    xy_coords = cos(two_pi * xy_coords / period_n_pix)

    # note the use of smooth_step to threshold cos output for smooth transition
    rgbs = make_tsr([1., 1., 1.]) - 1. * smooth_step(t_max(xy_coords, axis=1), 0.9, 1.0).unsqueeze(-1)
    return rgbs


def rotate_grid(xy_coords, canvas_wh, t):
    xy_coords = xy_coords - canvas_wh / 2  # centering

    t = (t + 1/16) * two_pi
    R = make_tsr([cos(t), -sin(t), sin(t), cos(t)]).reshape(2, 2)
    xy_coords = (R @ xy_coords.T).T

    return grid(xy_coords, canvas_wh, t)


def main():
    canvas_wh = (512, 256)

    entry(grid, canvas_wh, fname="out/02_grid")

    entry(rotate_grid, canvas_wh, num_frames=90, fps=30, fname="out/02_grid_rotate")


if __name__ == '__main__':
    main()