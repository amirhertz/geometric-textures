from custom_types import *
from constants import CACHE_ROOT, DATA_ROOT
from process_data import mesh_utils
import torch
import os
from options import Options
from process_data.mesh_plotter import plot_mesh
from process_data import files_utils


class MeshHandler:

    _mesh_dss: List[Union[mesh_utils.MeshDS, N]] = []
    _upsamplers: List[Union[mesh_utils.Upsampler, N]] = []

    def __init__(self, path_or_mesh: Union[str, T_Mesh], opt: Options, level: int, local_axes: Union[N, TS] = None):
        self.level = level
        self.opt = opt
        if type(path_or_mesh) is str:
            self.raw_mesh = mesh_utils.load_mesh(path_or_mesh)
        else:
            self.raw_mesh: T_Mesh = path_or_mesh
        self.update_ds(self.raw_mesh, level)
        self.valid_axes = local_axes is not None
        if local_axes is None:
            self.local_axes = self.extract_local_axes()
        else:
            self.local_axes = local_axes

    @staticmethod
    def reset():
        MeshHandler._mesh_dss = []
        MeshHandler._upsamplers = []

    # in place
    def to(self, device: D):
        self.raw_mesh = (self.vs.to(device), self.faces.to(device))
        self.ds.to(device)
        self.upsampler.to(device)
        self.local_axes = self.local_axes[0].to(device), self.local_axes[1].to(device)
        return self

    def detach(self):
        self.vs = self.vs.detach()
        self.local_axes = self.local_axes[0].detach(), self.local_axes[1].detach()
        return self

    @staticmethod
    def pad_ds(level: int):
        for i in range(len(MeshHandler._mesh_dss), level + 1):
            MeshHandler._mesh_dss.append(None)
            MeshHandler._upsamplers.append(None)

    def fill_ds(self, mesh: T_Mesh, level: int):
        MeshHandler._mesh_dss[level] = mesh_utils.MeshDS(mesh).to(mesh[0].device)
        MeshHandler._upsamplers[level] = mesh_utils.Upsampler(mesh).to(mesh[0].device)

    def update_ds(self, mesh: T_Mesh, level: int):
        self.pad_ds(level)
        if MeshHandler._mesh_dss[level] is None:
            self.fill_ds(mesh, level)

    # in place
    def upsample(self, in_place: bool = True):
        if not in_place:
            return self.copy().upsample()
        self.raw_mesh = self.upsampler(self.raw_mesh)
        self.vs = self.opt.scale_vs_factor * self.vs
        self.level += 1
        self.update_ds(self.raw_mesh, self.level)
        return self

    def __add__(self, deltas: T):
        mesh = self.copy()
        mesh += deltas
        return mesh

    def projet_displacemnets(self, deltas: T) -> T:
        _, local_axes = self.extract_local_axes()
        deltas = deltas.squeeze(0).t().reshape(-1, 3)
        global_vecs = torch.einsum('fsad,fa->fsd', [local_axes, deltas]).view(-1, 3)
        vs_deltas = global_vecs[self.ds.vertex2faces] * self.ds.vertex2faces_ma[:, :, None]
        vs_deltas = vs_deltas.sum(1) / self.ds.vs_degree[:, None]
        return vs_deltas

    def __iadd__(self, deltas: T):
        self.vs = self.vs + self.projet_displacemnets(deltas)
        return self

    @staticmethod
    def get_local_axes(mesh: T_Mesh) -> Tuple[T, T]:
        vs, faces = mesh
        _, normals = mesh_utils.compute_face_areas(mesh)
        vs_faces = vs[faces]
        origins = [((vs_faces[:, i] + vs_faces[:, (i + 1) % 3]) / 2).unsqueeze(1) for i in range(3)]
        x = [(vs_faces[:, (i + 1) % 3] - vs_faces[:, i]) for i in range(3)]
        y = [torch.cross(x[i], normals) for i in range(3)]  # 3 f 3
        axes = [torch.cat(list(map(lambda v: v.unsqueeze(1), [x[i], y[i], normals])), 1) for i in range(3)]
        axes = torch.cat(axes, dim=1).view(-1, 3, 3, 3)
        origins = torch.cat(origins, dim=1).view(-1, 3, 3)
        axes = axes / torch.norm(axes, p=2, dim=3)[:, :, :, None]
        return origins, axes

    def extract_local_axes(self) -> Tuple[T, T]:
        if self.valid_axes:
            return self.local_axes
        self.local_axes = self.get_local_axes(self.raw_mesh)
        self.valid_axes = True
        return self.local_axes

    def extract_features(self, z: Union[T, float], noise_before: bool) -> TS:

        def extract_local_cords() -> T:
            vs_ = self.vs
            if self.opt.noise_before and type(z) is T:
                if self.opt.fix_vs_noise:
                    vs_ = vs_ + z
                else:
                    vs_ += z
                if self.opt.update_axes:
                    self.vs = vs_
                    origins, local_axes = self.extract_local_axes()
                else:
                    origins, local_axes = self.get_local_axes((vs_, self.faces))
            else:
                origins, local_axes = self.extract_local_axes()
            global_cords = vs_[self.ds.face2points] - origins
            local_cords = torch.einsum('fsd,fsad->fsa', [global_cords, local_axes])
            return local_cords, vs_

        def get_edge_lengths() -> T:
            nonlocal vs
            vs_faces = vs[self.faces]
            lengths = list(map(lambda i: (vs_faces[:, (i + 1) % 3] - vs_faces[:, i]).norm(2, 1).unsqueeze(1), range(3)))
            lengths = torch.cat(lengths, 1)
            return lengths.unsqueeze(2)

        opposite_vs, vs = extract_local_cords()
        edge_lengths = get_edge_lengths()
        fe = torch.cat((opposite_vs, edge_lengths), 2).view(len(self), -1).t()
        if not noise_before:
            fe = fe + z
        # return torch.rand(1, 12, len(self), device=self.device)
        return fe.unsqueeze(0)

    def export(self, path):
        if not self.opt.debug:
            mesh_utils.export_mesh((self.vs / (self.opt.scale_vs_factor ** self.level), self.faces), path)

    def plot(self, save_path: str = '', ambient_color: T = T((255., 200, 255.)),
        light_dir: T = T((.5, .5, 1))):
        return plot_mesh(self.mesh_copy, save_path=save_path, ambient_color=ambient_color, light_dir=light_dir)

    @property
    def vs(self) -> T:
        return self.raw_mesh[0]

    @vs.setter
    def vs(self, vs_new):
        self.valid_axes = False
        self.raw_mesh = vs_new, self.faces

    @property
    def faces(self) -> T:
        return self.raw_mesh[1]

    @faces.setter
    def faces(self, faces_new):
        self.raw_mesh = self.vs, faces_new

    @property
    def device(self) -> D:
        return self.vs.device

    @property
    def ds(self) -> mesh_utils.MeshDS:
        return MeshHandler._mesh_dss[self.level]

    @property
    def upsampler(self) -> mesh_utils.Upsampler:
        return MeshHandler._upsamplers[self.level]

    @property
    def gfmm(self) -> T:
        return self.ds.gfmm

    @property
    def mesh_copy(self) -> T_Mesh:
        return self.vs.detach().cpu().clone(), self.faces.detach().cpu().clone()

    def __call__(self, z: Union[T, float] = 0, noise_before: bool = False) -> T:
        return self.extract_features(z, noise_before)

    def __len__(self) -> int:
        return self.faces.shape[0]

    def copy(self):
        mesh = self.vs.clone(), self.faces.clone()
        if self.valid_axes:
            local_axes = self.local_axes[0].clone(), self.local_axes[1].clone()
        else:
            local_axes = None
        return MeshHandler(mesh, self.opt, self.level, local_axes=local_axes)


class MeshInference(MeshHandler):

    def __init__(self, mesh_name: str, path_or_mesh: Union[str, T_Mesh], opt: Options, level: int,
                 local_axes: Union[N, TS] = None):
        self.mesh_name = mesh_name
        super(MeshInference, self).__init__(path_or_mesh, opt, level, local_axes)

    def grow_add(self, deltas: T):
        displacemnets = self.projet_displacemnets(deltas)
        self.vs = self.vs + self.projet_displacemnets(deltas)
        return self, displacemnets

    def fill_ds(self, mesh: T_Mesh, level: int):
        if not self.load(level):
            super(MeshInference, self).fill_ds(mesh, level)
            self.save(level)

    def load(self, level: int) -> bool:
        cache = []
        if files_utils.load_pickle(self.cache_path(level), cache):
            cache = cache[0]
            MeshHandler._mesh_dss[level] = cache['mesh_dss'].to(self.device)
            MeshHandler._upsamplers[level] = cache['upsamplers'].to(self.device)
            return True
        return False

    def save(self, level: int):
        files_utils.save_pickle({'mesh_dss': self._mesh_dss[level].to(CPU),
                                 'upsamplers': self._upsamplers[level].to(CPU)}, self.cache_path(level))
        self._mesh_dss[level].to(self.device)
        self._upsamplers[level].to(self.device)

    def cache_path(self, level: int):
        return f'{CACHE_ROOT}/{self.mesh_name}/{self.mesh_name}_{level:02d}.pkl'

    def copy(self):
        mesh = self.vs.clone(), self.faces.clone()
        if self.valid_axes:
            local_axes = self.local_axes[0].clone(), self.local_axes[1].clone()
        else:
            local_axes = None
        return MeshInference(self.mesh_name, mesh, self.opt, self.level, local_axes=local_axes)

def load_template_mesh(opt: Options, level) -> Tuple[str, T_Mesh]:
    mesh_path = f'{DATA_ROOT}/{opt.mesh_name}/{opt.mesh_name}_template.obj'
    if not os.path.isfile(mesh_path):
        return opt.template_name, mesh_utils.load_real_mesh(opt.template_name, level)
    else:
        mesh = mesh_utils.scale_mesh(mesh_path, False, opt.mesh_name, 0)
        if level > 0:
            mesh_handler = MeshHandler(mesh, opt, 0)
            for i in range(level):
                mesh_handler.upsample()
            mesh = mesh_handler.raw_mesh
        return f'{opt.mesh_name}_template', mesh
