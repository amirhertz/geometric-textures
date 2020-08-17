from custom_types import *
import torch.nn as nn
from models.single_mesh_conv import MultiMeshConv
import options
from models.mesh_handler import MeshHandler, MeshInference
import abc


class Model(nn.Module, abc.ABC):
    def __init__(self):
        super(Model, self).__init__()
        self.save_model: Union[N, Callable[[nn.Module]]] = None

    def save(self):
        self.save_model(self)


class SingleMeshModel(Model, abc.ABC):

    def __init__(self, opt: options.Options, is_generator: bool):
        super(SingleMeshModel, self).__init__()
        self.opt = opt
        self.levels, self.fe_lists = self.init_model(opt, is_generator)

    def get_level(self, level: int) -> nn.Module:
        return self.levels[level]

    def dup(self, level: int):
        if 0 < level < len(self.levels) and self.fe_lists[level] == self.fe_lists[level - 1]:
            self.levels[level].load_state_dict(self.levels[level - 1].state_dict())

    @staticmethod
    def num_fe(opt: options.Options, level: int, i: int, is_generator: bool) -> int:
        if i == 0:
            fe = opt.in_nf
        elif i == opt.num_layers - 1:
            fe = (opt.out_nf - 1) * int(is_generator) + 1
        else:
            fe = max(min(opt.start_nf * 2 ** level, opt.max_nf) // 2 ** (i - 1), opt.min_nf)
        return fe

    @staticmethod
    def init_model(opt: options.Options, is_generator: bool) -> Tuple[nn.ModuleList, List[List[int]]]:
        fe_lists = [[SingleMeshModel.num_fe(opt, level, i, is_generator) for i in range(opt.num_layers)]
                    for level in range(opt.num_levels)]
        levels = nn.ModuleList([MultiMeshConv(fe_list)
                                for fe_list in fe_lists])
        return levels, fe_lists


class SingleMeshGenerator(SingleMeshModel):

    def __init__(self, opt: options.Options):
        super(SingleMeshGenerator, self).__init__(opt, True)

    def forward_level(self, m: MeshHandler, z: Union[T, float], level: int) -> MeshHandler:
        features = m(z, self.opt.noise_before)
        m += self.levels[level](features, m.gfmm)
        return m

    def grow_forward(self, m: MeshInference, z: List[Union[T, float]], level: int, start_level: int) -> TS:

        def grow_level(z_, level_) -> T:
            nonlocal m
            features = m(z_, self.opt.noise_before)
            out = self.levels[level_](features, m.gfmm)
            m, displacement = m.grow_add(out)
            return displacement

        displacements: TS = []

        with torch.no_grad():
            for j, i in enumerate(range(start_level, level)):
                displacements.append(grow_level(z[j], i))
                m = m.upsample()
            displacements.append(grow_level(z[-1], level))
        return displacements

    def inference_forward(self, m: MeshHandler, z: List[Union[T, float]], level: int, start_level: int, path: str, upsample: bool = True) -> MeshHandler:
        with torch.no_grad():
            for j, i in enumerate(range(start_level, level)):
                m = self.forward_level(m, z[j], i)
                m.export(f'{path}_{start_level}{i}')
                if upsample:
                    m = m.upsample()
            m = self.forward_level(m, z[-1], level)
            m.export(f'{path}_{start_level}{level}')
            return m

    def forward(self, m: MeshHandler, z: List[Union[T, float]], level: int, start_level: int = 0, upsample: bool =True, exports=None) -> MeshHandler:
        with torch.no_grad():
            for j, i in enumerate(range(start_level, level)):
                m = self.forward_level(m, z[j], i)
                if upsample:
                    m = m.upsample()
        return self.forward_level(m, z[-1], level)


class SingleMeshDiscriminator(SingleMeshModel):

    def __init__(self, opt: options.Options):
        super(SingleMeshDiscriminator, self).__init__(opt, False)

    def forward(self, m: MeshHandler, level: int) -> T:
        return self.levels[level](m(), m.gfmm)

    def penalty_forward(self, m: MeshHandler, level: int, features: T) -> T:
        return self.levels[level](features, m.gfmm)