# import open3d as o3d
from io import BytesIO
from pathlib import Path

import k3d
import numpy as np
from numpy.linalg import inv
from PIL import Image
from .render import (
    camera_pose, unproject,
)

# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)


def rgb256_to_hex(rgb):
    """k3d requires color to be specified in hex"""
    # rgb: list of 3 numbers, each in [0,255]
    # return int('0x%02x%02x%02x' % (rgb[0], rgb[1], rgb[2]), 16)
    R, G, B = rgb
    ret = (R << 16) + (G << 8) + B
    return int(ret)  # make sure the final type is int


def img_to_bytes(img, format="PNG", **kwargs):
    with BytesIO() as f:
        img.save(f, format=format, **kwargs)
        return f.getvalue()


class OpenVisWrapper():
    """
    Please use this wrapper to set intrinsics / extrinsics on Open3D.
    It's fine that you want to directly call the APIs on the Visualizer,
    but please be very careful.

    For example, calling self.vis.change_field_of_view to modify intrinsics
    can only get you up to 90 degrees, at which point you are capped.
    Our set_intrinsics routine can bypass that restriction.

    Open3D's documentation quality and interaction code can have some
    inconsistencies. It's intended as a user-friendly visualizer. But
    since we are planning to do reconstruction on its renderings,
    we need to be precise.
    """
    def __init__(self, width=800, height=600):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=width, height=height)
        self.ctr = self.vis.get_view_control()
        self.opts = self.vis.get_render_option()
        self.opts.mesh_show_wireframe = True
        self.opts.line_width = 10.0

    def blocking_run(self):
        # this call blocks the main thread;
        self.vis.run()

    def to_image(self):
        # self.vis.capture_screen_image(str(fname), do_render=True)
        img = self.vis.capture_screen_float_buffer(do_render=True)
        img = np.asarray(img)
        return img

    def __del__(self):
        self.vis.destroy_window()

    def add(self, geom):
        self.vis.add_geometry(geom)
        self.vis.update_geometry(geom)

    def set_intrinsics(self, intrinsic):
        # ideally img w and h are not needed to set intrinsics.
        # but because of the way the open3d internals work, we have to provide it
        assert intrinsic.shape == (3, 4)
        intrinsic = self.resolve_intrinsic_inconsistency(intrinsic)
        self._hard_set_camera(intr=intrinsic)

    def set_extrinsics(self, ext):
        assert ext.shape == (4, 4)
        ext = self.resolve_extrinsic_inconsistency(ext)
        self._hard_set_camera(extr=ext)

    @staticmethod
    def resolve_intrinsic_inconsistency(intrinsic):
        # nobody cares about the sign of a windowing transform?
        intrinsic = intrinsic.copy()
        intrinsic[1, 1] = -intrinsic[1, 1]
        intrinsic = intrinsic[0:3, 0:3]
        return intrinsic

    @staticmethod
    def resolve_extrinsic_inconsistency(ext):
        ext = ext.copy()
        # HC: open3d (as of 22.01.10) does something bad when it comes to importing/exporting extrinsics
        # it seems intentional, but I don't understand why they do it
        # pipe the computed extrinsics matrix through this routine before feeding it to open3d

        # 1. back out the eye vector
        eye = -1 * inv(ext[0:3, 0:3]).dot(ext[0:3, 3])

        # 2. flip the up and front; open3d messes up here
        ext[1, 0:3] = -1 * ext[1, 0:3]
        ext[2, 0:3] = -1 * ext[2, 0:3]

        # 3. reproject the original eye, so that the error cancels out itself
        ext[0:3, 3] = -1 * ext[0:3, 0:3].dot(eye)

        return ext

    def _hard_set_camera(self, intr=None, extr=None):
        old_param = self.ctr.convert_to_pinhole_camera_parameters()
        new_param = o3d.camera.PinholeCameraParameters()
        new_param.extrinsic = extr if extr is not None else old_param.extrinsic
        new_param.intrinsic = old_param.intrinsic
        if intr is not None:
            new_param.intrinsic.intrinsic_matrix = intr
        self.ctr.convert_from_pinhole_camera_parameters(new_param, allow_arbitrary=True)


def generate_spiral_camera_trajectory(
    points, num_steps=20, num_rounds=1, height_multi=3, radius_mult=1.5
):
    points = np.asarray(points)
    bbox = compute_axis_aligned_bbox(points)
    height = bbox[1, 1] - bbox[1, 0]
    max_side = (bbox[:, 1] - bbox[:, 0]).max()
    center = (bbox[:, 1] + bbox[:, 0]) / 2

    radius = max_side * radius_mult
    height = height * height_multi
    eyes = []
    for i in range(num_steps):
        y_offset = (height / 2) - i * (height / num_steps)

        angle = i * (360 * num_rounds / num_steps)
        angle = angle / 180 * np.pi
        x_offset, z_offset = radius * np.array([np.cos(angle), np.sin(angle)])

        eyes.append(
            center + [x_offset, y_offset, z_offset]
        )

    poses = []
    for eye in eyes:
        # here the benefit of a loose "up" direction becomes obvious
        p = camera_pose(eye, center - eye, [0, 1, 0])
        poses.append(p)
    return poses


def compute_axis_aligned_bbox(xyz):
    """
        xyz: [n, d]
        return: [d, 2] where each row is (min, max)
    """
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    bbox = np.stack([mins, maxs], axis=1)
    return np.array(bbox)


def draw_camera(
    K, pose, img_w, img_h, scale=0.3, color=[0.8, 0.2, 0.8],
    top_left_corner=[-0.5, -0.5],
    return_raw=False
):
    tl_x, tl_y = top_left_corner
    corner_pixels = np.array([
        [tl_x, tl_y],  # top left
        [tl_x + img_w, tl_y],  # top right
        [tl_x + img_w, tl_y + img_h],  # bottom right
        [tl_x, tl_y + img_h],  # bottom left
    ])

    pts = unproject(K, corner_pixels, depth=scale)
    pts = np.concatenate([
        pts,
        np.array([0, 0, 0, 1]).reshape(1, -1),
    ], axis=0)  # origin, followed by 4 img corners
    pts = pts @ pose.T
    pts = pts[:, :3]

    assert pts.shape == (5, 3)

    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 0], [4, 1], [4, 2], [4, 3],
    ], dtype=int)

    if return_raw:
        return pts, lines

    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


class CameraCone():
    def __init__(
        self, K, pose, img_w, img_h, scale=0.3,
        top_left_corner=[-0.5, -0.5], color=[200, 40, 200],
    ):
        tl_x, tl_y = top_left_corner
        corner_pixels = np.array([
            [tl_x, tl_y],  # top left
            [tl_x + img_w, tl_y],  # top right
            [tl_x + img_w, tl_y + img_h],  # bottom right
            [tl_x, tl_y + img_h],  # bottom left
        ])

        pts = unproject(K, corner_pixels, depth=scale)
        pts = np.concatenate([
            pts,
            np.array([0, 0, 0, 1]).reshape(1, -1),
        ], axis=0)  # origin, followed by 4 img corners
        pts = pts @ pose.T
        pts = pts[:, :3]

        self.pts = pts.astype(np.float32)

        color = np.array(color)
        assert np.issubsctype(color.dtype, np.integer) \
            and color.min() >= 0 and color.max() < 256
        self.color = color

    def as_k3d_lineset(self):
        inds = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 0], [4, 1], [4, 2], [4, 3],
        ]

        # .tolist() so that dtype is not np.int64, but regular int
        # k3d demands regular int
        color = rgb256_to_hex(self.color.tolist())

        lines = k3d.lines(
            self.pts, inds, indices_type='segment', color=color
        )
        return lines

    def as_k3d_viewplane(self, fname_or_npimg):
        corner_pts = self.pts[:, :4]
        verts = corner_pts
        conns = [
            [0, 2, 1],
            [0, 3, 2],
        ]
        # note here uv is per-vertex; not face driven; strange
        # [0, 0] at top left of the texture img; my corners are ordered clockwise
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]

        if isinstance(fname_or_npimg, (str, Path)):
            # always read in, then convert to jpeg bytes
            img = np.array(Image.open(str(fname_or_npimg)))
        else:
            img = np.array(fname_or_npimg)
        del fname_or_npimg

        img = Image.fromarray(img)
        im_bytes = img_to_bytes(img, format="JPEG", quality=50)

        mesh = k3d.mesh(
            verts, conns, side="double", uvs=uvs,
            texture=im_bytes, texture_file_format="jpeg",
            # needed so that default blue is not blended into texture
            # this is a bug that the devs should fix
            color=0xffffff
        )
        return mesh

    def _as_line_set(self):
        raise NotImplementedError("deprecated; use k3d")
        lines = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 0], [4, 1], [4, 2], [4, 3],
        ], dtype=np.int32)
        print("warn: expecting float color in [0, 1]. But the API is changed to take [0, 255]")
        colors = np.array([self.color] * len(lines), dtype=np.float32)

        lset = o3d.t.geometry.LineSet()
        lset.point['positions'] = self.pts
        lset.line['indices'] = lines
        lset.line['colors'] = colors
        return lset

    def _as_view_plane(self, fname_or_npimg):
        # raise NotImplementedError("deprecated; use k3d")
        corner_pts = self.pts[:, :4]
        verts = corner_pts
        conns = np.array([
            [0, 2, 1],
            [0, 3, 2],
        ], dtype=np.int32)
        uvs = np.array([
            [(0, 1), (1, 0), (1, 1)],
            [(0, 1), (0, 0), (1, 0)],
        ], dtype=np.float32)  # note uv 0, 0 is at bottom left of img

        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex['positions'] = verts
        mesh.triangle['indices'] = conns
        mesh.triangle["texture_uvs"] = uvs

        mat = mesh.material
        mat.material_name = "defaultUnlit"

        img = None
        if isinstance(fname_or_npimg, str):
            img = o3d.t.io.read_image(fname_or_npimg)
        else:
            img = o3d.t.geometry.Image(fname_or_npimg)

        mat.texture_maps['albedo'] = img
        assert mat.is_valid()

        return mesh


def quick_vis_3d(*geoms):
    renderer = OpenVisWrapper()
    for obj in geoms:
        renderer.add(obj)
    renderer.blocking_run()


def o3d_pc(points, color=None):
    points = points[..., :3]
    n = len(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color:
        colors = [color for _ in range(n)]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
