import torch
from utils import entry, device, two_pi, make_tsr, cos, sin, t_max, smooth_step, step, fract

TIME_FACTOR = 50.


def grid_using_sine(xy_coords, canvas_wh, t):
    period_n_pix = 40 + 20 * sin(1.0 * t * two_pi)  # grid size itself changes over time

    xy_coords = cos(two_pi * xy_coords / period_n_pix)

    # note the use of smooth_step to threshold cos output for smooth transition
    rgbs = make_tsr([1., 1., 1.]) - 1. * smooth_step(0.9, 1.0, t_max(xy_coords, axis=1)).unsqueeze(-1)
    return rgbs


def rotate_grid(xy_coords, canvas_wh, t):
    xy_coords = xy_coords - canvas_wh / 2  # centering

    t = (t + 1/16) * two_pi
    R = make_tsr([cos(t), -sin(t), sin(t), cos(t)]).reshape(2, 2)
    xy_coords = (R @ xy_coords.T).T

    return grid_using_sine(xy_coords, canvas_wh, t)


def period_step(xy_coords, p):
    xy = step(p, fract(xy_coords))
    xs, ys = xy.T
    return xs, ys


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


def checker(xy_coords, canvas_wh, t):
    xy_coords = (2. * xy_coords - canvas_wh) / canvas_wh[1]
    xy_coords = xy_coords * (2. + t * TIME_FACTOR)
    xs, ys = period_step(xy_coords, 0.5)
    colors = apply_smooth_logic(xs, ys, "xor")
    rgbs = make_tsr([1.0, 1.0, 1.0]) * colors.unsqueeze(-1)
    return rgbs


def grid(xy_coords, canvas_wh, t):
    xy_coords = (2. * xy_coords - canvas_wh) / canvas_wh[1]
    xy_coords = xy_coords * (2. + t * TIME_FACTOR)
    xs, ys = period_step(xy_coords, 0.05)
    colors = apply_smooth_logic(xs, ys, "and")
    rgbs = make_tsr([1.0, 1.0, 1.0]) * colors.unsqueeze(-1)
    return rgbs


def integral_01_oscillate(x, p=0.5):
    return torch.floor(x) * (1 - p) + torch.relu(fract(x) - p)


def integrated_period_step(xy_coords, dx_dy, p):
    integrated_xys = (
        integral_01_oscillate(xy_coords + dx_dy, p) - integral_01_oscillate(xy_coords, p)
    ) / dx_dy
    xs, ys = integrated_xys.T
    return xs, ys


def checker_antialias(xy_coords, canvas_wh, t):
    xy_coords = (2. * xy_coords - canvas_wh) / canvas_wh[1]
    xy_coords = xy_coords * (2. + t * TIME_FACTOR)

    # dx and dy must go through the same change
    dx_dy = make_tsr([1., 1.])
    dx_dy = (2. * dx_dy) / canvas_wh[1]  # NOTE: subtraction does NOT affect dx, dy
    dx_dy = dx_dy * (2. + t * TIME_FACTOR)

    xs, ys = integrated_period_step(xy_coords, dx_dy, 0.5)
    colors = apply_smooth_logic(xs, ys, "xor")
    rgbs = make_tsr([1.0, 1.0, 1.0]) * colors.unsqueeze(-1)
    return rgbs


def grid_antialias(xy_coords, canvas_wh, t):
    xy_coords = (2. * xy_coords - canvas_wh) / canvas_wh[1]
    xy_coords = xy_coords * (2. + t * TIME_FACTOR)

    # dx and dy must go through the same change
    dx_dy = make_tsr([1., 1.])
    dx_dy = (2. * dx_dy) / canvas_wh[1]  # NOTE: subtraction does NOT affect dx, dy
    dx_dy = dx_dy * (2. + t * TIME_FACTOR)

    xs, ys = integrated_period_step(xy_coords, dx_dy, 0.05)
    colors = apply_smooth_logic(xs, ys, "and")
    rgbs = make_tsr([1.0, 1.0, 1.0]) * colors.unsqueeze(-1)
    return rgbs


def main():
    canvas_wh = (512, 256)

    entry(grid_using_sine, canvas_wh, fname="out/02_grid_using_sine")

    entry(rotate_grid, canvas_wh, num_frames=90, fps=30, fname="out/02_grid_rotate")

    entry(checker, canvas_wh, num_frames=180, fps=30, fname="out/02_alias_checker")

    entry(grid, canvas_wh, num_frames=180, fps=30, fname="out/02_alias_grid")

    entry(checker_antialias, canvas_wh, num_frames=180, fps=30, fname="out/02_antialias_checker")

    entry(grid_antialias, canvas_wh, num_frames=180, fps=30, fname="out/02_antialias_grid")


if __name__ == '__main__':
    main()