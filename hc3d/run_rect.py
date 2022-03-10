from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .render import as_homogeneous, homogenize

root = Path("./data")


def ortho(xs):
    # for 3d, cross-product is good.
    # for d > 3, use qr factorization. much easier to write this way
    # xs: [n, 3]
    n, _ = xs.shape
    x1, x2, x3 = xs.T
    zeros = np.zeros(n)
    cross_product_mats = np.array([
        [zeros, -x3, x2],
        [x3, zeros, -x1],
        [-x2, x1, zeros]
    ])
    cross_product_mats = np.transpose(cross_product_mats, (2, 0, 1))
    return cross_product_mats


def compute_homography(pts1, pts2):
    assert pts1.shape == pts2.shape
    assert pts1.shape[1] == 2
    n = pts1.shape[0]
    pts1, pts2 = as_homogeneous(pts1), as_homogeneous(pts2)

    pts2 = ortho(pts2)  # [n, 3, 3]
    A = np.einsum("nab, nc -> nabc", pts2, pts1)
    A = A.reshape(3 * n, -1)

    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H1to2 = L.reshape(3, 3)
    return H1to2


def _compute_homography(pts1, pts2):
    assert pts1.shape == pts2.shape
    assert pts1.shape[1] == 2
    n = pts1.shape[0]

    # this is based on the slides and books. But it's hard to remember
    A = []
    for i in range(n):
        x, y = pts1[i]
        u, v = pts2[i]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)

    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H1to2 = L.reshape(3, 3)
    return H1to2


def load_corners():
    with open(root / "corners.json", "r") as f:
        corners = json.load(f)
        print(corners)

    points = np.array([
        corners['LL'],
        corners['LR'],
        corners['UR'],
        corners['UL']
    ])
    return points


def warp_image(img, H, target_w, target_h):
    # compared to opencv's warp, this routine does not fill in holes
    h, w = img.shape[:2]
    orig_ys, orig_xs = np.meshgrid(range(h), range(w), indexing="ij")
    orig_ys = orig_ys.reshape(-1)
    orig_xs = orig_xs.reshape(-1)

    coords = np.stack([
        orig_xs, orig_ys, np.ones(h * w)
    ], axis=0)
    coords = H @ coords
    coords = coords[:2] / coords[2]
    coords = np.rint(coords).astype(int)  # round to integers
    xs, ys = coords

    canvas = np.zeros((target_h, target_w, 3))  # background color black

    # throw out those beyond image boundary
    inbound_mask = (ys < target_h) & (ys >= 0) & (xs < target_w) & (xs >= 0)
    ys, xs = ys[inbound_mask], xs[inbound_mask]
    orig_ys, orig_xs = orig_ys[inbound_mask], orig_xs[inbound_mask]

    canvas[ys, xs] = img[orig_ys, orig_xs]
    return canvas


def rectify_image():
    img = np.array(Image.open(root / "stadium.png")) / 255.

    pts1 = load_corners()
    aspect_ratio = 1 / 1.88

    new_height = 500
    new_width = new_height * aspect_ratio

    tl_x = 300
    tl_y = 500
    pts2 = np.array([
        [tl_x, tl_y],
        [tl_x + new_width, tl_y],
        [tl_x + new_width, tl_y + new_height],
        [tl_x, tl_y + new_height],
    ])
    pts2 = np.rint(pts2)

    H = compute_homography(pts1, pts2)
    what = as_homogeneous(pts1) @ H.T
    what = np.rint(homogenize(what)).astype(int)
    assert (what[:, :2] == pts2).all()

    warped = warp_image(img, H, 1000, 1200)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[0].set_title("original")
    axes[1].imshow(warped)
    axes[1].set_title("warped")
    plt.show()


def show_field_corners():
    img = np.array(Image.open(root / "stadium.png"))
    points = load_corners()

    segments = []
    for i in range(4):
        start, end = points[i], points[(i + 1) % 4]
        segments.append([start, end])

    plt.imshow(img)
    line_segments = LineCollection(segments, linestyle='solid', color="cyan")
    ax = plt.gca()
    ax.add_collection(line_segments)
    plt.show()


def main():
    pass
    # show_field_corners()
    rectify_image()


if __name__ == "__main__":
    main()
