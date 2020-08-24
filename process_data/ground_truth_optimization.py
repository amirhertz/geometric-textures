from process_data import files_utils, mesh_utils
from custom_types import *
from options import GtOptions
import pickle
import os
import constants as const
from models.factory import Logger
from models.chamfer import ChamferLoss


class GroundTruthGenerator:

    @staticmethod
    def gravity_loss(mesh: T_Mesh, ds: mesh_utils.VerticesDS) -> T:
        vs, _ = mesh
        vs_ne = vs[ds.vertex2vertex] * ds.vertex2faces_ma[:, :, None]
        diff = (vs - vs_ne.sum(1) / ds.vs_degree[:, None]).abs()
        gravity_loss = diff.sum(1).mean()
        return gravity_loss

    @staticmethod
    def local_edge_entropy(mesh: T_Mesh) -> T:
        vs, faces = mesh
        vs_faces = vs[faces]
        lengths = list(map(lambda i: (vs_faces[:, (i + 1) % 3] - vs_faces[:, i]).norm(2, 1).unsqueeze(1), range(3)))
        lengths = torch.cat(lengths, 1)
        lengths /= lengths.sum(1)[:, None]
        negative_entropy = (lengths * torch.log(lengths)).sum(1)
        return negative_entropy.mean()

    @property
    def refinement(self) -> bool:
        return self.opt.num_levels == 0

    def __init__(self, opt: GtOptions, device: D):
        self.opt = opt
        self.level = self.opt.start_level
        self.device = device
        self.chamfer = ChamferLoss(device)
        self.scales = self.load_scales()
        self.target_mesh: Union[T_Mesh, N] = None
        self.target_transform = None
        self.source_mesh: Union[T_Mesh, N] = None
        self.ds_source: Union[mesh_utils.VerticesDS, N] = None
        self.logger = Logger(self.opt)

    def update_target(self):
        if (len(self.opt.switches) == 0 or self.opt.switches[-1] < self.level) and self.target_mesh is None:
            self.target_mesh = mesh_utils.load_mesh(f'{const.RAW_MESHES}/{self.opt.mesh_name}')
            self.target_mesh, self.target_transform = mesh_utils.to_unit_cube(self.target_mesh)

        elif self.level == 0:
            self.target_mesh = mesh_utils.load_mesh(f'{const.RAW_MESHES}/{self.opt.mesh_name}_0')
            self.target_mesh, self.target_transform = mesh_utils.to_unit_cube(self.target_mesh)

        elif self.level in self.opt.switches:
            self.target_mesh = mesh_utils.load_mesh(f'{const.RAW_MESHES}/{self.opt.mesh_name}_{self.level}')
            self.target_mesh = (self.target_mesh[0] - self.target_transform[0][None, :]) * self.target_transform[1], self.target_mesh[1]
            self.opt.gamma_edge_global = self.opt.gamma_edge_local = 0
        self.target_mesh = mesh_utils.to(self.target_mesh, self.device)

    def adjust_edge_scale(self) -> T_Mesh:
        if len(self.scales) == 0:
            mean_length = mesh_utils.edge_lengths(self.source_mesh,self.ds_source.edges_ind).mean().item()
            self.scales.append(1 / mean_length)
        else:
            self.scales.append(self.scales[0] * 2 ** self.level)
        return self.source_mesh[0] / (2 ** self.level), self.source_mesh[1]

    def triangulation_iter(self):

        def global_edge_loss():
            return mesh_utils.edge_lengths(self.source_mesh, self.ds_source.edges_ind).std()

        def local_edge_loss():
            return self.local_edge_entropy(self.source_mesh)

        def vs_loss():
            return self.gravity_loss(self.source_mesh, self.ds_source)

        loss = 0
        to_log = []
        for gamma, loss_func, name in zip(self.opt.triangulation_weights, [global_edge_loss, local_edge_loss, vs_loss],
                                          ['e_g', 'e_l', 'gravity',]):
            if gamma != 0:
                cur_loss = loss_func()
                loss += gamma * cur_loss
                to_log += [name, cur_loss]
        return loss, to_log

    def cp_iter(self, mesh: T_Mesh, *args):
        mcp_s2t, mcp_t2s = args
        distances_s2t = mcp_s2t(self.target_mesh, mesh[0]).mean()
        distances_t2s = mcp_t2s(mesh, self.target_mesh[0]).mean()
        loss = self.opt.gamma_distance_s2t * distances_s2t + self.opt.gamma_distance_t2s * distances_t2s
        return loss, ('d_s2t', distances_s2t, 'd_t2s', distances_t2s)

    def ch_iter(self):
        sampled_source = mesh_utils.sample_on_mesh(self.source_mesh, self.opt.num_samples[self.level])
        # sampled_target = mesh_utils.sample_on_sphere(self.target_mesh[0], self.opt.num_samples[self.level], self.device)
        sampled_target = mesh_utils.sample_on_mesh(self.target_mesh, self.opt.num_samples[self.level])
        chamfer_loss = self.chamfer(sampled_source, sampled_target)
        ch_loss = 0
        to_log = []
        for gamma, loss, name in zip(self.opt.chamfer_weights,
                                     chamfer_loss, ['dis_s2t', 'dis_t2s', 'n_s2t', 'n_t2s']):
            if gamma != 0:
                ch_loss += gamma * loss
                to_log += [name, loss]

        return ch_loss, to_log

    def optimize(self) -> T_Mesh:
        # mcp_s2t = MeshClosestPoint(self.target_mesh, self.source_mesh[0], self.ds_target).to(self.device)
        # mcp_t2s = MeshClosestPoint(self.source_mesh, self.target_mesh[0], self.ds_source).to(self.device)
        self.source_mesh[0].requires_grad = True
        optimizer = Optimizer([self.source_mesh[0]], lr=self.opt.lr)
        for i in range(self.opt.level_iters[self.level]):
            for iters, optim_func in zip([self.opt.triangulation_iters, self.opt.ch_iters], [self.triangulation_iter,
                                                                                             self.ch_iter]):
                for _ in range(iters):
                    optimizer.zero_grad()
                    loss, log = optim_func()
                    if type(loss) is not int:
                        loss.backward()
                        optimizer.step()
                    self.logger.stash_iter(*log)
            self.logger.reset_iter()
        return (self.source_mesh[0].detach(), self.source_mesh[1])

    def load_scales(self) -> List:
        if os.path.isfile(f'{self.opt.cp_folder}/{self.opt.mesh_name}_scales.pkl'):
            with open(f'{self.opt.cp_folder}/{self.opt.mesh_name}_scales.pkl', 'rb') as f:
                scales = pickle.load(f)
            scales = scales[:self.level]
        else:
            scales = []
        return scales

    def save_scales(self):
        with open(f'{self.opt.cp_folder}/{self.opt.mesh_name}_scales.pkl', 'wb') as f:
            pickle.dump(self.scales, f, pickle.HIGHEST_PROTOCOL)

    def between_levels(self):
        self.logger.start()
        self.update_target()
        if self.source_mesh is None:
            last_mesh_path = f'{self.opt.cp_folder}/{self.opt.mesh_name}_level{self.level - 1 + self.refinement:02d}.obj'
            if self.level > 0 and os.path.isfile(last_mesh_path):
                self.source_mesh = mesh_utils.to(mesh_utils.load_mesh(last_mesh_path), self.device)
                self.source_mesh = self.source_mesh[0] * (2 ** self.level), self.source_mesh[1]
                self.target_mesh = self.target_mesh[0] * (2 ** self.level), self.target_mesh[1]
            else:
                mesh_path = f'{const.RAW_MESHES}/{self.opt.mesh_name}_template.obj'
                if not os.path.isfile(mesh_path) or not self.opt.pre_template:
                    # mesh_path = f'{const.RAW_MESHES}/icosahedron.obj'
                    self.source_mesh = mesh_utils.load_real_mesh(self.opt.template_name, self.opt.template_start, False)
                else:
                    self.source_mesh = mesh_utils.load_mesh(mesh_path)
                self.source_mesh = mesh_utils.to(mesh_utils.to_unit_cube(self.source_mesh)[0], self.device)
                scale = 1 / mesh_utils.edge_lengths(self.source_mesh).mean().item()
                self.target_mesh = self.target_mesh[0] * scale, self.target_mesh[1]
                self.source_mesh = self.source_mesh[0] * scale, self.source_mesh[1]
                self.target_transform = self.target_transform[0], self.target_transform[1] * scale

                if self.opt.template_start > 0 or os.path.isfile(mesh_path):
                    mesh_utils.export_mesh(self.source_mesh, f'{self.opt.cp_folder}/{self.opt.mesh_name}_template.obj')
        if self.level != 0 and not self.refinement:
            self.source_mesh = mesh_utils.Upsampler(self.source_mesh)(self.source_mesh)
        self.ds_source = mesh_utils.VerticesDS(self.source_mesh).to(self.device)

        self.save_scales()

    def generate_ground_truth_meshes(self):
        files_utils.init_folders(self.opt.cp_folder + '/')
        for i in range(self.opt.start_level, self.opt.start_level + self.opt.num_levels + self.refinement):
            self.between_levels()
            self.source_mesh = self.optimize()
            scaled_mesh = self.adjust_edge_scale()
            self.source_mesh = self.source_mesh[0] * 2, self.source_mesh[1]
            self.target_mesh = self.target_mesh[0] * 2, self.target_mesh[1]
            self.target_transform = self.target_transform[0], self.target_transform[1] * 2
            mesh_utils.export_mesh(scaled_mesh, f'{self.opt.cp_folder}/{self.opt.mesh_name}_level{self.level:02d}.obj')
            self.logger.stop(False)
            self.level += 1
        self.save_scales()


if __name__ == '__main__':
    device_ = CUDA(0)
    opt_ = GtOptions()
    args = GtOptions(online_demo=True)
    args.fill_args(tag='demo', mesh_name='cloud', template_name='sphere', num_levels=6)
    gt_gen = GroundTruthGenerator(args, CUDA(0))
    # gt_gen = GroundTruthGenerator(opt_, device_)
    # gt_gen.generate_ground_truth_meshes()
