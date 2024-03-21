"""
this is a pytorch imitation of the shadertoy programming environment for procedural graphics / image generation
"""
import torch
from utils import entry, device, two_pi, make_tsr, cos, coords_take, mix, smooth_step


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

    # horizontal period 0.7 (debugs that x grows from left to right); as t goes, shift [right to left] by 3 periods
    tick = xs / 0.7 + t * 3
    rgbs[:, 0] = 0.5 + 0.5 * cos(tick * two_pi)
    return rgbs


def artsy_sine(xy_coords, canvas_wh, t):
    xy_coords = xy_coords / canvas_wh
    xyx = coords_take(xy_coords, "xyx")
    rgbs = 0.5 + 0.5 * cos(10 * t + xyx + make_tsr([0, 2, 4]))
    return rgbs


"""
vec3 rulerColor(float t) {
    t = clamp(log(t+1.0), 0.0, 1.0);
    return mix(
        mix(vec3(0., .1., 1.), vec3(1., .1, 0.), t*5.), 
        vec3(1.0), 
        smoothstep(.2, .5, t)
    );
}
"""
def special(xy_coords, canvas_wh, t):
    xy_coords = xy_coords / canvas_wh
    xs, ys = xy_coords.T
    ts = xs

    # ts expected to be positive
    ts = torch.clamp(torch.log(1.0 + ts), 0.0, 1.0)
    # ts = torch.clamp(ts, 0.0, 1.0)

    def _m(*args):
        return make_tsr(args).unsqueeze(-1)

    colors = mix(
        mix(_m(0.0, 0.1, 1.0), _m(1.0, 0.1, 0.0), ts * 20.),
        _m(1.0, 1.0, 1.0),
        smooth_step(0.2, 0.5, ts)
    )
    return colors.T


def main():
    # canvas_wh = (256, 128)

    # entry(vertical_sine_wave, canvas_wh, fname="out/01_vertical_sine")

    # entry(horizontal_sine_wave, canvas_wh, num_frames=90, fps=30, fname="out/01_horizontal_sine")

    # canvas_wh = (512, 256)
    # entry(artsy_sine, canvas_wh, num_frames=90, fps=30, fname="out/01_artsy")

    canvas_wh = (256, 256)
    entry(special, canvas_wh, fname="out/01_special_color")


if __name__ == '__main__':
    main()
