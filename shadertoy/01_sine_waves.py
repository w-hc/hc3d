"""
this is a pytorch imitation of the shadertoy programming environment for procedural graphics / image generation
"""
import torch
from utils import entry, device, two_pi, make_tsr, cos, coords_take


def dummy_shader(xy_coords, canvas_wh, t):
    # [n, 2] where each is (x, y) coord with origin at top-left
    return 0.5 * torch.ones(len(xy_coords), 3, device=device)


def vertical_sine_wave(xy_coords, canvas_wh, t):
    xy_coords = xy_coords / canvas_wh
    xs, ys = xy_coords.T

    rgbs = torch.zeros(len(xy_coords), 3, device=device)
    # period 0.7; debug that y grows from top to bottom
    rgbs[:, 0] = 0.5 + 0.5 * cos((ys / 0.7) * two_pi)
    return rgbs


def horizontal_sine_wave(xy_coords, canvas_wh, t):
    xy_coords = xy_coords / canvas_wh
    xs, ys = xy_coords.T
    rgbs = torch.zeros(len(xy_coords), 3, device=device)

    # horizontal period 0.7 (debugs that x grows from left to right); as t goes, shift right by 3 periods
    tick = xs / 0.7 + t * 3
    rgbs[:, 0] = 0.5 + 0.5 * cos(tick * two_pi)
    return rgbs


def artsy_sine(xy_coords, canvas_wh, t):
    xy_coords = xy_coords / canvas_wh
    xyx = coords_take(xy_coords, "xyx")
    rgbs = 0.5 + 0.5 * cos(10 * t + xyx + make_tsr([0, 2, 4]))
    return rgbs


def main():
    canvas_wh = (256, 128)

    entry(vertical_sine_wave, canvas_wh, fname="out/01_vertical_sine")

    entry(horizontal_sine_wave, canvas_wh, num_frames=90, fps=30, fname="out/01_horizontal_sine")

    canvas_wh = (512, 256)
    entry(artsy_sine, canvas_wh, num_frames=90, fps=30, fname="out/01_artsy")


if __name__ == '__main__':
    main()
