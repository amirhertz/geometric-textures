from custom_types import *
import options as options
from process_data import mesh_utils
import models.factory as factory
from models.single_mesh_models import SingleMeshGenerator
from models.mesh_handler import MeshHandler, MeshInference, load_template_mesh


class DGTS:

    def __init__(self, opt: Union[options.Options, options.TrainOption], device: D):
        self.opt = opt.load()
        self.generator: SingleMeshGenerator = factory.model_lc(opt, SingleMeshGenerator, device=device)
        self.level = 0
        self.device = device

    def get_random_z(self, num_randoms: int) -> T:
        if self.opt.noise_before:
            return torch.randn(num_randoms, 3, device=self.device) * self.opt.noise_amplitude
        return torch.randn(1, self.generator.opt.in_nf, num_randoms, device=self.device) * self.opt.noise_amplitude

    def get_z_by_level(self, base_mesh: MeshHandler, level: int) -> T:
        num_faces = len(base_mesh) * 4 ** level
        if self.opt.noise_before:
            num_faces = (num_faces - len(base_mesh)) // 2 + base_mesh.vs.shape[0]
        return self.get_random_z(num_faces)

    def get_z_sequence(self, base_mesh: MeshHandler, max_level: int) -> TS:
        return [self.get_z_by_level(base_mesh, level) for level in range(max_level + 1)]

    def __len__(self):
        return len(self.generator.levels)


class Mesh2Mesh(DGTS):

    def __init__(self, opt: options.Options, device: D):
        super(Mesh2Mesh, self).__init__(opt, device)
        self.generator.eval()

    def trim(self, start: int, end: int) -> Tuple[int, int]:
        if end < 0 or end > len(self) - 1:
                end = len(self) - 1
        if start > end:
            start = end
        return start, end

    def get_z_sequence(self, base_mesh: MeshHandler, max_level: int) -> factory.Noise:
        return factory.Noise(data=super(Mesh2Mesh, self).get_z_sequence(base_mesh, max_level))

    def growing(self, mesh: MeshInference, start: int, end: int, num_frames: int, zero_places: NoiseT = ()) ->  factory.Noise:

        export_name = f'{self.opt.cp_folder}/inference/{mesh.mesh_name}/scene00'

        start, end = self.trim(start, end)
        if len(zero_places) == 1:
            zero_places = zero_places * (end - start + 1)
        z = self.get_z_sequence(mesh, end - start)
        for i in range(min(len(z), len(zero_places))):
            if zero_places[i]:
                z[i] = 0
        deltas = self.generator.grow_forward(mesh.copy(), z, end, start)
        mesh.export(f'{export_name}/{0:02d}')
        for i in range(end - start + 1):
            base_vs, cur_delta = mesh.vs.clone(), deltas[i]
            for j in range(num_frames):
                mesh.vs = base_vs + cur_delta * (j + 1) / num_frames
                mesh.export(f'{export_name}/{(num_frames * i + j + 1):02d}')
                print(f'done: {num_frames * i + j + 1}/{num_frames * (end - start + 1)}')
            if i < end - start:
                mesh.upsample()
        return z

    def animate(self, mesh: MeshInference, start: int, end: int, num_scene: int, num_frames: Tuple[int, int], zero_places: NoiseT = ()):
        export_name = f'{self.opt.cp_folder}/inference/{mesh.mesh_name}/scene01'
        start, end = self.trim(start, end)
        if len(zero_places) == 1:
            zero_places = zero_places * (end - start + 1)
        z_a = self.growing(mesh.copy(), start, end, num_frames[0], zero_places)
        z_b = self.get_z_sequence(mesh, end - start).to(self.device)
        for i in range(min(len(z_a), len(zero_places))):
            if zero_places[i]:
                z_b[i] = 0
        z_start = z_a
        num_frames = num_frames[1]
        for s in range(num_scene):
            for i in range(num_frames):
                alpha = (i + 1) / float(num_frames)
                z = z_a * (1 - alpha) + z_b * alpha
                m = mesh.copy()
                out = self.generator(m, z, end, start, upsample=True)
                out.export(f'{export_name}/{s * num_frames + i:02d}')
                print(f'frame {s * num_frames + i + 1} / {num_scene * num_frames}...')

            z_a = z_b
            if s == num_scene - 2:
                z_b = z_start
            else:
                z_b = self.get_z_sequence(mesh, end - start).to(self.device)

    def __call__(self, mesh: Union[str, MeshHandler, T_Mesh], start: int, end: int, zero_places: NoiseT = 0) -> MeshHandler:
        MeshHandler.reset()
        start, end = self.trim(start, end)
        if type(zero_places) is int:
            zero_places = [zero_places]
        if len(zero_places) == 1:
            zero_places = zero_places * (end - start + 1)
        if mesh is None:
            mesh = MeshHandler(mesh_utils.load_real_mesh(self.opt.template_name, start),self.opt, 0).to(self.device)
        elif type(mesh) is not MeshHandler:
            mesh = MeshHandler(mesh, self.opt, 0).to(self.device)
        z = self.get_z_sequence(mesh, end - start)
        for i in range(min(len(z), len(zero_places))):
            if zero_places[i]:
                z[i] = 0
        remeshed = self.generator.forward(mesh, z, end, start, upsample=True)
        return remeshed


class MeshGen(DGTS):

    def __init__(self, opt: options.Options, device: D):
        super(MeshGen, self).__init__(opt, device)
        self.generator.eval()
        template_name, template = load_template_mesh(opt, opt.start_level)
        self.template = MeshInference(template_name, template, self.opt, self.opt.start_level).to(self.device)
        self.reconstruction_z = factory.NoiseMem(opt).load().to(device)

    def compose_z(self, start_level) -> factory.Noise:
        random_noise = self.get_z_sequence(self.template, len(self) - 1)
        noise = self.reconstruction_z[: start_level] + random_noise[start_level:]
        return noise

    def generate_seq(self, num_seqs: int):
        for seq in range(num_seqs):
            z = self.compose_z(0)
            self.generator.inference_forward(self.template.copy(), z, len(self) - 1, 0,
                                             f'{opt_.cp_folder}/inference/gen/{self.opt.mesh_name}_{seq}',
                                             upsample=True)

    def generate_all(self, num_samples: int):
        for i in range(len(self.generator.levels)):
            for j in range(num_samples):
                out_mesh = self(i)
                out_mesh.export(f'{self.opt.cp_folder}/inference/gen/{self.opt.mesh_name}_{i}_{j:02d}')
                print(f'gen {self.opt.mesh_name} {i * num_samples + j +1:02d} / {len(self.generator.levels) * num_samples}')

    def __call__(self, start_level: int):
        with torch.no_grad():
            if start_level < 0:
                start_level = len(self)
            start_level = min(len(self), start_level)
            z = self.compose_z(start_level)
            return self.generator(self.template.copy(), z, len(self) - 1)


if __name__ == '__main__':
    opt_ = options.Options()
    opt_.parse_cmdline()
    device = CPU
    with_noise = False
    if opt_.gen_mode == 'generate':
        mg = MeshGen(opt_, device)
        mg.generate_all(opt_.num_gen_samples)
    elif opt_.gen_mode == 'animate':
        m2m = Mesh2Mesh(opt_, device)
        in_mesh = MeshInference(opt_.target, mesh_utils.load_real_mesh(opt_.target, 0, True), opt_, 0).to(device)
        m2m.animate(in_mesh, opt_.gen_levels[0], opt_.gen_levels[1], 0, (12, 17), zero_places=(0, 0, 1, 1, 1))



