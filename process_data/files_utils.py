import os
import constants as const
import time
import pickle
from typing import List


def init_folders(*folders):
    if not const.DEBUG:
        for f in folders:
            dir_name = os.path.dirname(f)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)


def load_pickle(path: str, pointer: List) -> bool:
    path = add_suffix(path, '.pkl')
    if os.path.isfile(path):
        with open(path, 'rb') as f:
             pointer.append(pickle.load(f))
        return True
    return False


def save_pickle(obj, path: str):
    path = add_suffix(path, '.pkl')
    init_folders(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def collect(root:str, *suffix, prefix='') -> list:
    paths = []
    if not os.path.isdir(root):
        print(f'Warning: trying to collect from {root} but dir isn\'t exist')
    else:
        p_len = len(prefix)
        for path, _, files in os.walk(root):
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                p_len_ = min(p_len, len(file_name))
                if file_extension in suffix and file_name[:p_len_] == prefix:
                    paths.append((path, file_name, file_extension))
        paths.sort(key=lambda x: os.path.join(x[1], x[2]))
    return paths


def add_suffix(path: str, suffix: str) -> str:
    if len(path) < len(suffix) or path[-len(suffix):] != suffix:
        path = f'{path}{suffix}'
    return path


def measure_time(func, num_iters: int, *args):
    start_time = time.time()
    for i in range(num_iters):
        func(*args)
    total_time = time.time() - start_time
    avg_time = total_time / num_iters
    print(f"{str(func).split()[1].split('.')[-1]} total time: {total_time}, average time: {avg_time}")
