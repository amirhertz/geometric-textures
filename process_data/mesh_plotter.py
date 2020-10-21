from custom_types import *
from process_data import mesh_utils, files_utils
import matplotlib as mpl
import matplotlib.pylab as pl
import mpl_toolkits.mplot3d as a3
from PIL import Image
import os
mpl.use('Agg')


def init_plot(mesh):
    figure = pl.figure()
    ax = figure.add_subplot(111, projection='3d')
    # hide axis, thank to
    # https://stackoverflow.com/questions/29041326/3d-plot-with-matplotlib-hide-axes-but-keep-axis-labels/
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    vs = mesh[0]
    lim = [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf]
    for i in range(3):
        lim[2 * i] = min(lim[2 * i], vs[:, i].min())
        lim[2 * i + 1] = max(lim[2 * i], vs[:, i].max())
    return (ax, lim, figure)


def add_surfaces(mesh: V_Mesh, plot, rgb: V):
    vs, faces = mesh
    vtx = vs[faces]
    tri = a3.art3d.Poly3DCollection(vtx, facecolors=rgb)
    plot[0].add_collection3d(tri)
    return plot

def compute_colors(mesh: T_Mesh, ambient_color: T, light_dir: T) -> V:
    _, normals = mesh_utils.compute_face_areas(mesh)
    light_dir = light_dir / light_dir.norm(2, 0)
    colors = torch.einsum('fd,d,r->fr', normals, light_dir, ambient_color) / 510. + .5
    return colors.numpy()


# in place
def fix_vertices(mesh: T_Mesh, scale_by: Union[N, float] = None) -> float:
    vs, _ = mesh
    z = -vs[:, 2].clone()
    vs[:, 2] = vs[:, 1]
    vs[:, 1] = z
    max_range = 0
    for i in range(3):
        min_value = vs[:, i].min().item()
        max_value = vs[:, i].max().item()
        max_range = max(max_range, max_value - min_value)
        vs[:, i] -= min_value
    if scale_by is None:
        scale_by = max_range
    vs /= scale_by
    return scale_by


def fig2data(plot) -> V:
    # taken from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    # Thanks!
    canvas = plot[2].canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf.copy()


def data2image(np_image: V):
    return Image.fromarray(np_image)


def render_mesh(mesh: Union[T_Mesh, str], ambient_color: T, light_dir: T):
    if type(mesh) is str:
        mesh = mesh_utils.load_mesh(mesh)
    _ = fix_vertices(mesh)
    colors = compute_colors(mesh, ambient_color, light_dir)
    mesh = mesh[0].numpy(), mesh[1].numpy()
    plot = init_plot(mesh)
    add_surfaces(mesh, plot, colors)
    li = max(plot[1][1], plot[1][3], plot[1][5])
    plot[0].auto_scale_xyz([0, li], [0, li], [0, li])
    pl.tight_layout()
    fig_data = fig2data(plot)
    pl.close(plot[2])
    return fig_data


def blend_images(images: List[V], blend_height: int, blend_width: int, rows: int) -> List[V]:
    cols = len(images) // rows
    for i in range(cols - 1):
        for j in range(rows):
            image_index = i + j * cols
            blend_a = images[image_index][:, -blend_width:]
            blend_b = images[image_index + 1][:, : blend_width]
            ma = blend_b < blend_a
            blend_a[ma] = blend_b[ma]
            images[image_index][:, -blend_width:] = blend_a
            images[image_index + 1] = images[image_index + 1][:, blend_width:]
    for i in range(rows - 1):
        for j in range(cols):
            image_index = i * cols + j
            blend_a = images[image_index][-blend_width:, :]
            blend_b = images[image_index + cols][: blend_width, :]
            ma = blend_b < blend_a
            blend_a[ma] = blend_b[ma]
            images[image_index][-blend_width:, :] = blend_a
            images[image_index + cols] = images[image_index + cols][blend_width:, :]
    return images


def make_pretty(np_images: List[V], offset=(.15, .1, .3, .1), blend=0.15, rows=1):
    if type(offset) is not tuple:
        offset = [offset] * 4
    offset = [- np_images[0].shape[idx % 2] if off == 0 else int(np_images[0].shape[idx % 2] * off) for idx, off in enumerate(offset)]
    cols = len(np_images) // rows
    np_images = np_images[: cols * rows]

    # offset_height, offset_width = int(np_images[0].shape[0] * offset ), int(np_images[0].shape[1] * offset)
    blend_height, blend_width = int(np_images[0].shape[0] * blend), int(np_images[0].shape[1] * blend)
    np_images = [image[offset[3]: - offset[1], offset[0]: - offset[2]] for image in np_images]
    # np_images = [image[offset_height: - offset_height, offset_width: - offset_width] for image in np_images]
    if blend != 0:
        np_images = blend_images(np_images, blend_height, blend_width, rows)
    np_images = [np.concatenate(np_images[i * cols: (i + 1) * cols], axis=1) for i in range(rows)]
    im = np.concatenate(np_images, axis=0)
    return data2image(im)


def plot_mesh(*meshes: Union[T_Mesh, str], save_path: str = '',
              ambient_color: T = T((255., 200, 255.)), light_dir: T = T((.5, .5, 1))):
    np_images = [render_mesh(mesh, ambient_color, light_dir) for mesh in meshes]
    im = make_pretty(np_images)
    if save_path:
        save_path = files_utils.add_suffix(save_path, '.png')
        files_utils.init_folders(save_path)
        im.save(save_path)
    return im
