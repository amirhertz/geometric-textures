from custom_types import *
import numpy as np
from process_data.files_utils import init_folders, add_suffix
import os
import pickle
import constants as const


class Upsampler:

    def __init__(self, template: T_Mesh):
        vs_t, faces_t = template
        base_inds = 3 * torch.arange(faces_t.shape[0], device=vs_t.device).unsqueeze(1) + vs_t.shape[0]
        raw_edges = torch.cat([faces_t[:, [i, (i + 1) % 3]] for i in range(3)], dim=1).view(-1, 2).sort()[0].cpu()
        mask = torch.ones(raw_edges.shape[0], dtype=torch.bool)
        mapper = torch.zeros(raw_edges.shape[0], dtype=torch.int64)
        edges = dict()
        counter = 0
        for i, edge in enumerate(raw_edges):
            e_ = (edge[0].item(), edge[1].item())
            if e_ in edges:
                mask[i] = 0
                mapper[i] = edges[e_]
            else:
                mapper[i] = counter
                edges[e_] = counter
                counter += 1
        faces_mid = [(torch.arange(3 * faces_t.shape[0], device=vs_t.device) + vs_t.shape[0]).view(-1, 3)]
        faces_sr = [torch.cat([faces_t[:, i].unsqueeze(1), base_inds + i, base_inds + (i + 2) % 3], dim=1) for i in
                    range(3)]
        mapper = torch.cat((torch.arange(vs_t.shape[0]), mapper + vs_t.shape[0])).to(device=vs_t.device)
        self.down_faces = faces_t
        self.up_faces = mapper[torch.cat(faces_sr + faces_mid, dim=0)]
        self.mask = mask.to(vs_t.device)
        self.down_len = vs_t.shape[0]

    def __call__(self, mesh: T_Mesh) -> T_Mesh:
        vs, faces = mesh
        mid_points = torch.cat([torch.mean(vs[faces][:, [i, (i + 1) % 3], :], dim=1) for i in range(3)],
                               dim=1).view(-1, 3)
        up_vs = torch.cat((vs, mid_points[self.mask.to(vs.device)]), dim=0)
        return up_vs, self.up_faces

    def down_sample(self, mesh: T_Mesh) -> T_Mesh:
        vs, faces = mesh
        return vs[: self.down_len], self.up_faces

    def to(self, device: D):
        self.up_faces = self.up_faces.to(device)
        self.mask = self.mask.to(device)
        self.down_faces = self.down_faces.to(device)
        return self


class MeshDS: # : mesh data structures

    MAX_V_DEG = 0

    def __init__(self, mesh: T_Mesh):
        if self.MAX_V_DEG == 0:
            self.init_v_degree(mesh)
        vs, faces = to(mesh, CPU) # seems to work faster on cpu
        self.gfmm = torch.zeros(3, faces.shape[0], dtype=torch.int64)
        self.vertex2faces = torch.zeros(vs.shape[0], self.MAX_V_DEG, dtype=torch.int64) - 1
        self.__vertex2faces_flipped = None
        self.vs_degree = torch.zeros(vs.shape[0], dtype=torch.int64)
        self.build_ds(faces)
        self.vertex2faces_ma = (self.vertex2faces > -1).float()
        self.face2points = self.get_face2points(faces)

    # inplace
    def to(self, device: D):
        self.gfmm = self.gfmm.to(device)
        self.vertex2faces = self.vertex2faces.to(device)
        self.vs_degree = self.vs_degree.to(device)
        self.face2points = self.face2points.to(device)
        self.vertex2faces_ma = self.vertex2faces_ma.to(device)
        return self

    @property
    def vertex2faces_flipped(self) -> T:
        if self.__vertex2faces_flipped is None:
            mod = self.vertex2faces % 3
            flipped = self.vertex2faces - mod + (mod - 1) % 3
            self.__vertex2faces_flipped = flipped
        return self.__vertex2faces_flipped.to(self.gfmm.device)

    def update_vs_degree(self, face, face_id, zero_one_two):
        self.vertex2faces[face, self.vs_degree[face]] = face_id * 3 + zero_one_two
        self.vs_degree[face] += 1

    def build_ds(self, faces):

        def insert_edge():
            nonlocal edges_count

            if edge not in edge2key:
                edge_key = edges_count
                edge2key[edge] = edge_key
                edge2faces[edge2key[edge], 0] = face_id
                edges_count += 1
            else:
                edge_key = edge2key[edge]
                edge2faces[edge_key, 1] = face_id
            edge2key_cache[face_id * 3 + idx] = edge_key

        def insert_face():
            nb_faces = edge2faces[edge2key_cache[face_id * 3 + idx]]
            nb_face = nb_faces[0] if nb_faces[0] != face_id else nb_faces[1]
            self.gfmm[nb_count[face_id], face_id] = nb_face
            nb_count[face_id] += 1

        edge2key = dict()
        edge2key_cache = torch.zeros(int(faces.shape[0] * 3), dtype=torch.int64)
        edges_count = 0
        edge2faces = torch.zeros(int(faces.shape[0] * 1.5), 2, dtype=torch.int64)
        nb_count = torch.zeros(self.gfmm .shape[1], dtype=torch.int64)
        zero_one_two = torch.arange(3)
        for face_id, face in enumerate(faces):
            faces_edges = [(face[i].item(), face[(i + 1) % 3].item()) for i in range(3)]
            self.update_vs_degree(face, face_id, zero_one_two)
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))
                insert_edge()
        for face_id, face in enumerate(faces):
            for idx in range(3):
                insert_face()
        self.vs_degree = self.vs_degree.float()

    def get_face2points(self, faces) -> T:
        cords_indices = torch.zeros(len(self), 3, dtype=torch.int64)
        all_inds = faces[self.gfmm.t()]
        for i in range(3):
            ma_a = all_inds - faces[:, i][:, None, None]
            ma_b = all_inds - faces[:, (i + 1) % 3][:, None, None]
            ma = (ma_a * ma_b) == 0
            ma_final = (ma.sum(2) == 2)[:, :, None] * (~ma)
            cords_indices[:, i] = all_inds[ma_final]
        return cords_indices

    @staticmethod
    def init_v_degree(mesh: T_Mesh):
        vs, faces = mesh
        vs_degree = torch.zeros(vs.shape[0], dtype=torch.int64)
        for face in faces:
            vs_degree[face] += 1
        MeshDS.MAX_V_DEG = max(vs_degree.max().item(), 6)

    def __len__(self):
        return self.gfmm.shape[1]


class VerticesDS(MeshDS):

    def __init__(self, mesh: T_Mesh):
        mesh = mesh[0].detach().cpu(), mesh[1].detach().cpu()
        super(VerticesDS, self).__init__(mesh)
        self.vertex2vertex = self.compute_vertex2vertex(mesh)
        self.faces_ring = self.compute_faces_ring(mesh)
        self.edges_ind = get_edges_ind(mesh)

    def update_vs_degree(self, face: T, face_id: int, _):
        self.vertex2faces[face, self.vs_degree[face]] = face_id
        self.vs_degree[face] += 1

    def compute_vertex2vertex(self, mesh: T_Mesh) -> T:
        vs, faces = mesh
        raw = faces[self.vertex2faces]
        base_inds = torch.arange(vs.shape[0])
        ma = base_inds[:, None, None] != raw
        wh = torch.where(~ma)
        wh = wh[0], wh[1], (wh[2] + 1) % 3
        ma[wh] = False
        ma[~self.vertex2faces_ma.bool()] = torch.tensor((True, False, False))
        v2v = raw[ma].view(vs.shape[0], self.vertex2faces.shape[1])
        return v2v

    def compute_faces_ring(self, mesh) -> T:
        vs, faces = mesh
        faces_ring = self.vertex2faces[faces]
        # ma_self = faces_ring == torch.arange(faces.shape[0], device=device)[:, None, None]
        # ma_self[:, 0, :] = False
        # ma_nb = faces_ring == self.gfmm.t()[:, :, None]
        # faces_ring[ma_self + ma_nb] = -1
        return faces_ring.view(faces.shape[0], -1)

    def to(self, device: D):
        super(VerticesDS, self).to(device)
        self.vertex2vertex = self.vertex2vertex.to(device)
        self.faces_ring = self.faces_ring.to(device)
        self.edges_ind = self.edges_ind.to(device)
        return self

def to(mesh: T_Mesh, device: D) -> T_Mesh:
    return (mesh[0].to(device), mesh[1].to(device))


def tetrahedron(callback=lambda x: x):
    y_0 = 1 / (2 * np.cos(np.pi / 6))
    y_1 = y_0 / 2
    z = np.sqrt(2) * y_0
    vs = [
        [0, y_0, 0],
        [.5, - y_1, 0],
        [-.5, - y_1, 0],
        [0, 0, z]
    ]
    fs = [
        [0, 1, 2],
        [1, 0, 3],
        [2, 1, 3],
        [0, 2, 3]
    ]
    return callback((T(vs).float(), T(fs).long()))


def compute_face_areas(mesh: T_Mesh) -> T_Mesh:
    vs, faces = mesh
    face_normals = torch.cross(vs[faces[:, 1]] - vs[faces[:, 0]], vs[faces[:, 2]] - vs[faces[:, 1]])
    face_areas = torch.norm(face_normals, p=2, dim=1)
    face_normals = face_normals / face_areas[:, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


def get_edges_ind(mesh: T_Mesh) -> T:
    vs, faces = mesh
    raw_edges = torch.cat([faces[:, [i, (i + 1) % 3]] for i in range(3)]).sort()
    raw_edges = raw_edges[0].cpu().numpy()
    edges = {(int(edge[0]), int(edge[1])) for edge in raw_edges}
    edges = torch.Tensor(list(edges)).long().to(faces.device)
    return edges


def edge_lengths(mesh: T_Mesh, edges_ind: TN = None) -> T:
    vs, faces = mesh
    if edges_ind is None:
        edges_ind = get_edges_ind(mesh)
    edges = vs[edges_ind]
    return torch.sqrt(((edges[:, 0] - edges[:, 1]) ** 2).sum(1))


# in place
def to_unit_cube(mesh: T_Mesh) -> V_Mesh:
    vs, _ = mesh
    max_vals = vs.max(0)[0]
    min_vals = vs.min(0)[0]
    max_range = (max_vals - min_vals).max() / 2
    center = (max_vals + min_vals) / 2
    vs -= center[None, :]
    vs /= max_range
    return mesh, (center, 1. / max_range)


# def cener_of_mass(mesh: V_Mesh) -> V:
#     vs, faces_ind = mesh
#     faces = vs[faces_ind]
#     volumes = (faces[:, 0, 0] * faces[:, 1, 1] * faces[:, 2, 2] - faces[:, 0, 0] * faces[:, 2, 1] * faces[:, 1, 2] -
#                faces[:, 1, 0] * faces[:, 0, 1] * faces[:, 2, 2] + faces[:, 1, 0] * faces[:, 2, 1] * faces[:, 0, 2] +
#                faces[:, 2, 0] * faces[:, 0, 1] * faces[:, 1, 2] - faces[:, 2, 0] * faces[:, 1, 1] * faces[:, 1, 2]) / 6
#     centers = faces.sum(1) * volumes[:, None] / 4
#     return centers.sum(0) / volumes.sum()


# in place
def to_unit_sphere(mesh: T_Mesh) -> T_Mesh:
    vs, _ = mesh
    # center = cener_of_mass(mesh)
    center = vs.mean(0)
    vs -= center[None, :]
    max_norm = torch.norm(vs, p=2, dim=1).max()
    vs /= max_norm
    return mesh


# in place
def to_unit_edge(mesh: T_Mesh) -> V_Mesh:
    vs, _ = mesh
    center = vs.mean(0)
    vs -= center[None, :]
    vs /= edge_lengths(mesh).mean()
    return mesh


def load_meshes(*files) -> List[T_Mesh]:
    return [load_mesh(file_name) for file_name in files]


def export_mesh(mesh: Union[V_Mesh, T_Mesh], file_name: str):
    vs, faces = to(mesh, CPU)
    file_name = add_suffix(file_name, '.obj')

    init_folders(file_name)
    if faces is not None:
        faces = faces + 1
    with open(file_name, 'w') as f:
        for vi, v in enumerate(vs):
            f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        if faces is not None:
            for face_id in range(faces.shape[0] - 1):
                f.write("f %d %d %d\n" % (faces[face_id][0], faces[face_id][1], faces[face_id][2]))
            f.write("f %d %d %d" % (faces[-1][0], faces[-1][1], faces[-1][2]))


def sample_on_mesh(mesh: T_Mesh, num_samples: int, face_areas: TN = None, normals: TN = None) -> TS:
    vs, faces = mesh
    if face_areas is None:
        face_areas, normals = compute_face_areas(mesh)
    face_areas = face_areas.detach()
    faces_ids = np.random.choice(range(face_areas.shape[0]), size=num_samples, p=(face_areas.cpu().numpy() / face_areas.sum().item()))
    chosen_faces = faces[torch.from_numpy(faces_ids).to(vs.device)]
    u, v = torch.rand(num_samples, 1, device=vs.device), torch.rand(num_samples, 1, device=vs.device)
    mask = u + v > 1
    u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
    w = 1 - u - v
    samples = u * vs[chosen_faces[:, 0]] + v * vs[chosen_faces[:, 1]] + w * vs[chosen_faces[:, 2]]
    return samples, normals[faces_ids]


# inplace
def flip_mesh(mesh: T_Mesh, inplace: bool = True) -> T_Mesh:
    vs, faces = mesh
    swap = faces[:, 0].clone()
    if not inplace:
        faces = faces.clone()
    faces[:, 0] = faces[:, 1]
    faces[:, 1] = swap
    return (vs, faces)


def load_mesh(file_name: str) -> T_Mesh:

    def off_parser():
        header = None

        def parser_(clean_line: list):
            nonlocal header
            if not clean_line:
                return False
            if len(clean_line) == 3 and not header:
                header = True
            elif len(clean_line) == 3:
                return 0, 0, float
            elif len(clean_line) > 3:
                return 1, -int(clean_line[0]), int

        return parser_

    def obj_parser(clean_line: list):
        if not clean_line:
            return False
        elif clean_line[0] == 'v':
            return 0, 1, float
        elif clean_line[0] == 'f':
            return 1, 1, int

    def fetch(lst: list, idx: int, dtype: type):
        if '/' in lst[idx]:
            lst = [item.split('/')[0] for item in lst[idx:]]
            idx = 0
        face_vs_ids = [dtype(c.split('/')[0]) for c in lst[idx:]]
        assert (len(face_vs_ids) == 3)
        return face_vs_ids

    def load_from_txt(parser) -> tuple:
        mesh_ = [[], []]
        with open(file_name, 'r') as f:
            for line in f:
                clean_line = line.strip().split()
                info = parser(clean_line)
                if not info:
                    continue
                mesh_[info[0]].append(fetch(clean_line, info[1], info[2]))

        mesh_ = [T(mesh_[0]).float(), T(mesh_[1]).long()]
        if mesh_[1].min() != 0:
            mesh_[1] -= 1
        return tuple(mesh_)
    for suffix in ['.obj', '.off']:
        file_name_tmp = add_suffix(file_name, suffix)
        if os.path.isfile(file_name_tmp):
            file_name = file_name_tmp
            break
    name, extension = os.path.splitext(file_name)
    if extension == '.obj':
        mesh = load_from_txt(obj_parser)
    elif extension == '.off':
        mesh = load_from_txt(off_parser())
    else:
        raise ValueError(f'mesh file {file_name} is not supported')
    assert ((mesh[1] >= 0) * (mesh[1] < len(mesh[0]))).all()
    return mesh


def scale_mesh(mesh_path: str, raw: bool, mesh_name: str, level: int) -> T_Mesh:
    scale = 0
    scales_path = f'{const.DATA_ROOT}/{mesh_name}/{mesh_name}_scales.pkl'
    if not raw and os.path.isfile(scales_path):
        with open(scales_path, 'rb') as f:
            scale = pickle.load(f)[level]
    mesh = load_mesh(mesh_path)
    if scale:
        mesh = mesh[0] * scale, mesh[1]
    elif raw:
        mesh = to_unit_edge(mesh)
    return mesh


def load_real_mesh(mesh_name: str, level: int, raw: bool = False) -> T_Mesh:
    if raw:
        mesh_path = f'{const.RAW_MESHES}/{mesh_name}'
    else:
        mesh_path = f'{const.DATA_ROOT}/{mesh_name}/{mesh_name}_level{level:02d}.obj'
    return scale_mesh(mesh_path, raw, mesh_name, level)


def sample_on_sphere(radius: float, num_samples: int, device: D = CPU) -> TS:
    theta = 2 * np.pi * torch.rand(num_samples)
    phi = torch.acos(1 - 2 * torch.rand(num_samples))
    sin_phi = torch.sin(phi)
    x = (sin_phi * torch.cos(theta)).unsqueeze(1)
    y = (sin_phi * torch.sin(theta)).unsqueeze(1)
    z = torch.cos(phi).unsqueeze(1)
    normals = torch.cat((x, y, z), dim=1).to(device)
    return radius * normals, normals


