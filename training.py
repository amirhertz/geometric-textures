from custom_types import *
import options as options
from torch import nn
import models.factory as factory
from models.single_mesh_models import SingleMeshDiscriminator
from models.mesh_handler import MeshHandler, load_template_mesh
from dgts_base import DGTS
from process_data import mesh_utils


class Trainer(DGTS):

    def __init__(self, opt: options.TrainOption, device: D):
        super(Trainer, self).__init__(opt, device)
        self.discriminator: SingleMeshDiscriminator = factory.model_lc(opt, SingleMeshDiscriminator, device=device)
        self.optimizer_generator: Union[factory.OptimizerLC, N] = None
        self.optimizer_discriminator: Union[factory.OptimizerLC, N] = None
        self.template = MeshHandler(load_template_mesh(opt, opt.start_level)[1],
                                    self.opt, self.opt.start_level).to(self.device)
        self.real_mesh: Union[MeshHandler, N] = None
        self.real_mesh_flipped: Union[MeshHandler, N] = None
        self.mse = nn.MSELoss().to(device)
        self.reconstruction_z = factory.NoiseMem(opt).load().to(device)
        self.logger = factory.Logger(opt)

    def get_z(self, reconstruction_mode: bool) -> List[Union[T, float]]:
        if reconstruction_mode and self.level >= self.opt.reconstruction_start:
            for i in range(len(self.reconstruction_z), self.level + 1):
                self.reconstruction_z.append(0. if i else self.get_z_by_level(self.template, 0).detach())
            return self.reconstruction_z
        else:
            return self.get_z_sequence(self.template, self.level)

    def generate(self, reconstruction_mode: bool, inside_out: bool = False) -> MeshHandler:
        z = self.get_z(reconstruction_mode)
        template = self.template.copy()
        if inside_out:
            template = template.flip()
        return self.generator(template, z, self.level)

    def before_level(self) -> [Optimizer, Optimizer, MeshHandler]:
        self.logger.start()
        self.generator.dup(self.level)
        self.discriminator.dup(self.level)
        self.optimizer_generator = factory.OptimizerLC(self.opt, 'generator', self.generator.get_level(self.level))
        self.optimizer_discriminator = factory.OptimizerLC(self.opt, 'discriminator', self.discriminator.get_level(self.level))
        self.real_mesh = MeshHandler(mesh_utils.load_real_mesh(self.opt.mesh_name,self.opt.start_level + self.level),
                                    self.opt, self.opt.start_level + self.level).to(self.device)
        if self.opt.inside_out:
            self.real_mesh_flipped = self.real_mesh.copy().flip()

    def between_levels(self):
        self.reconstruction_z.save()
        self.generator.save()
        self.discriminator.save()
        self.opt.save()
        self.logger.stop()
        self.level += 1

    def gradient_penalty(self, fake_mesh: MeshHandler) -> T:
        fake_data = fake_mesh().data
        real_data = self.real_mesh().data
        alpha = torch.rand(1, device=self.device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.discriminator.penalty_forward(self.real_mesh, self.level, interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return gradient_penalty.mean()

    def train_level(self):

        def penalty_iter():
            self.optimizer_discriminator.zero_grad()
            fake_mesh = self.generate(False, False).detach()
            gradient_penalty = self.gradient_penalty(fake_mesh)
            (self.opt.penalty_weight * gradient_penalty).backward(retain_graph=True)
            self.optimizer_discriminator.step()

        def discriminator_iter(inside_out: bool = False):
            if inside_out and not self.opt.inside_out:
                return
            self.optimizer_discriminator.zero_grad()
            fake_mesh = self.generate(False, inside_out).detach()
            real_mesh = self.real_mesh_flipped if inside_out else self.real_mesh
            out_real = self.discriminator(real_mesh.copy(), self.level)
            error_real = out_real.mean()
            out_fake = self.discriminator(fake_mesh, self.level)
            error_fake = out_fake.mean()
            gradient_penalty = self.gradient_penalty(fake_mesh)
            (self.opt.penalty_weight * gradient_penalty).backward(retain_graph=True)
            (error_fake - error_real).backward()
            self.optimizer_discriminator.step()
            self.logger.stash_iter('d_fake', error_fake, 'd_real', error_real)

        def generator_iter(inside_out: bool = False):
            if inside_out and not self.opt.inside_out:
                return
            nonlocal meshes
            self.optimizer_generator.zero_grad()
            fake_mesh = self.generate(False, inside_out)
            rec_mesh = self.generate(True, inside_out)
            out_fake = self.discriminator(fake_mesh, self.level)
            error_fake = out_fake.mean()
            error_rec = self.mse(rec_mesh.vs, self.real_mesh.vs)
            rec_weight = self.opt.reconstruction_weight
            fake_loss = - error_fake
            (rec_weight * error_rec + fake_loss).backward()
            self.optimizer_generator.step()
            self.logger.stash_iter('g_fake', error_fake, 'g_rec', error_rec)
            meshes = rec_mesh, fake_mesh

        def train_iter():
            for _ in range(self.opt.discriminator_iters):
                discriminator_iter()
            for _ in range(self.opt.generator_iters):
                generator_iter()
            self.logger.reset_iter()

        def decay(self):
            self.optimizer_generator.decay()
            self.optimizer_discriminator.decay()

        def mesh_paths(level: int, iteration: int) -> List[List[str]]:
            return [[f'{self.opt.cp_folder}/{export_type}/{tag}_{level:02d}_{iteration + 1:04d}' for
                    export_type in ['generated', 'plots']] for tag in ['rec', 'fake']]

        def plot(meshes: Tuple[MeshHandler, MeshHandler], paths: List[List[str]]):
            for mesh, path in zip(meshes, paths):
                mesh.export(path[0])
                mesh.plot(path[1])

        meshes: Union[Tuple[MeshHandler, ...], N] = None
        for iteration in range(self.opt.level_iters[self.level + self.opt.start_level]):
            train_iter()
            factory.do_when_its_time(self.opt.lr_decay_every, decay, iteration, self)
            factory.do_when_its_time(self.opt.export_meshes_every, plot, iteration, meshes, mesh_paths(self.level, iteration))

    def train(self):
        for _ in range(self.opt.num_levels):
            self.before_level()
            self.train_level()
            self.between_levels()


if __name__ == '__main__':
    opt_ = options.TrainOption()
    opt_.parse_cmdline()
    opt_ = opt_.load()
    device_ = CUDA(0)
    trainer = Trainer(opt_, device_)
    trainer.train()
