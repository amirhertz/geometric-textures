from custom_types import *
import torch.nn as nn
from models.single_mesh_models import SingleMeshGenerator, SingleMeshDiscriminator
from process_data.files_utils import init_folders
import shutil
import pickle
from options import Options, TrainOption, backward_compatibility
import os
from tqdm import tqdm


SingleModel = Union[SingleMeshGenerator, SingleMeshDiscriminator]


def is_model_clean(model: nn.Module) -> bool:
    for wh in model.parameters():
        if torch.isnan(wh).sum() > 0:
            return False
    return True


def model_lc(opt: Options, model_class: Type[SingleModel], device=CPU) -> SingleModel:

    def save_model(model_: SingleModel):
        nonlocal already_init, last_recover, model_path
        if opt.debug:
            return False
        if not already_init:
            init_folders(model_path)
            with open(params_path, 'wb') as f_p:
                pickle.dump({'params': opt, 'model_name': model_class_name}, f_p, pickle.HIGHEST_PROTOCOL)
            already_init = True
        if is_model_clean(model_):
            recover_path = f'{opt.cp_folder}/{model_class_name}_recover{last_recover + 1}.pth'
            if os.path.isfile(recover_path):
                os.remove(recover_path)
            torch.save(model_.state_dict(), model_path)
            shutil.copy(model_path, recover_path)
            last_recover = 1 - last_recover
            return True
        else:
            return False

    already_init = False
    model_class_name = str(model_class).split('.')[-1][:-2]
    last_recover = 0
    model_path = f'{opt.cp_folder}/{model_class_name}.pth'
    params_path = f'{opt.cp_folder}/{model_class_name}.pkl'
    if os.path.isfile(params_path):
        with open(params_path, 'rb') as f:
            model_params = pickle.load(f)
            opt = backward_compatibility(model_params['params'])
            model_class_name = model_params['model_name']
        already_init = True
    model = eval(model_class_name)(opt).to(device)
    if os.path.isfile(model_path):
        print(f'loading {model_class_name} model from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f'init {model_class_name} model')
    model.save_model = save_model
    return model


class OptimizerLC(Optimizer):

    def __init__(self, opt: TrainOption, optimizer_name: str, *models: nn.Module, device=CPU):
        self.already_init = False
        self.optimizer_path = f'{opt.cp_folder}/{optimizer_name}_optimizer.pkl'
        lr = opt.lr
        if type(lr) is float:
            lr = [lr] * len(models)
        super(OptimizerLC, self).__init__([
            {'params': model.parameters(), 'lr': lr[i], 'betas': opt.betas} for i, model in enumerate(models)])
        if os.path.isfile(self.optimizer_path):
            self.load_state_dict(torch.load(self.optimizer_path, map_location=device))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self, opt.lr_decay)

    def save(self):
        if not self.already_init:
            init_folders(self.optimizer_path)
            self.already_init = True
        torch.save(self.state_dict(), self.optimizer_path)

    def decay(self):
        self.scheduler.step()


def do_when_its_time(when, do, now, *with_what, default_return=None):
    if (now + 1) % when == 0:
        return do(*with_what)
    else:
        return default_return


class Noise(list):
    def __init__(self, **kwargs):
        super(Noise, self).__init__()
        if 'data' in kwargs:
            self.__iadd__(kwargs['data'])



    def __mul__(self, other: float):
        new_noise = Noise()
        for i in range(len(self)):
            new_noise.append(self[i] * other)
        return new_noise

    def __add__(self, other):
        new_noise = Noise()
        for i in range(min(len(other), len(self))):
            new_noise.append(self[i] + other[i])
        return new_noise

    def __rtruediv__(self, other: float):
        return self * (1 / other)

    def __rmul__(self, other: float):
        return self * other

    def to(self, device: D):
        for i in range(len(self)):
            if type(self[i]) is T:
                self[i] = self[i].to(device)
        return self


class NoiseMem(Noise):
    def __init__(self, opt: Options, **kwargs):
        super(Noise, self).__init__(**kwargs)
        self.save_path = f'{opt.cp_folder}/noise.pkl'
        self.debug: bool = opt is None or opt.debug

    def save(self):
        if not self.debug and len(self) > 0:
            to_save_list = []
            for i in range(len(self)):
                cur_noise = self[i]
                if type(cur_noise) is T:
                    cur_noise = cur_noise.clone().data.cpu()
                to_save_list.append(cur_noise)
            with open(self.save_path, 'wb') as f:
                pickle.dump(to_save_list, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        if os.path.isfile(self.save_path):
            with open(self.save_path, 'rb') as f:
                noise_list = pickle.load(f)
            if type(noise_list) is list:
                self.clear()
                self.__iadd__(noise_list)
        return self

class Logger:

    def __init__(self, opt: TrainOption):
        self.opt = opt
        self.level_dictionary = dict()
        self.iter_dictionary = dict()
        self.level = opt.start_level
        self.progress: Union[N, tqdm] = None

    @staticmethod
    def aggregate(dictionary: dict, parent_dictionary: Union[dict, N] = None) -> dict:
        aggregate_dictionary = dict()
        for key in dictionary:
            if 'counter' not in key:
                aggregate_dictionary[key] = dictionary[key] / float(dictionary[f"{key}_counter"])
                if parent_dictionary is not None:
                    Logger.stash(parent_dictionary, (key,  aggregate_dictionary[key]))
        return aggregate_dictionary

    @staticmethod
    def stash(dictionary: dict, items: Tuple[Union[str, Union[T, float]], ...]) -> dict:
        for i in range(0, len(items), 2):
            key, item = items[i], items[i + 1]
            if type(item) is T:
                item = item.item()
            if key not in dictionary:
                dictionary[key] = 0
                dictionary[f"{key}_counter"] = 0
            dictionary[key] += item
            dictionary[f"{key}_counter"] += 1
        return dictionary

    def stash_iter(self, *items: Union[str, Union[T, float]]):
        self.iter_dictionary = self.stash(self.iter_dictionary, items)

    def stash_level(self, *items: Union[str, Union[T, float]]):
        self.level_dictionary = self.stash(self.level_dictionary, items)

    def reset_iter(self):
        aggregate_dictionary = self.aggregate(self.iter_dictionary, self.level_dictionary)
        self.progress.set_postfix(aggregate_dictionary)
        self.progress.update()
        self.iter_dictionary = dict()

    @property
    def status_bar(self) -> tqdm:
        return tqdm(total=self.opt.level_iters[self.level - self.opt.start_level],
                    desc=f'{self.opt.name} Level: {self.level}')

    def start(self):
        self.progress = self.status_bar

    def stop(self, aggregate: bool = True):
        if aggregate:
            aggregate_dictionary = self.aggregate(self.level_dictionary)
            self.progress.set_postfix(aggregate_dictionary)
        self.level_dictionary = dict()
        self.progress.close()
        self.level += 1

    def reset_level(self, aggregate: bool = True):
        self.stop(aggregate)
        self.start()
