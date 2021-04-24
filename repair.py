import argparse
import constants as const
from custom_types import CUDA, CPU
from dgts_base import Mesh2Mesh
from process_data.ground_truth_optimization import GroundTruthGenerator
from process_data import mesh_utils
import sys
from training import Trainer
import options
import os

def str2bool(v):
  return v.lower() in ('true', '1')

parser = argparse.ArgumentParser()
parser.add_argument("train_mesh", type=str, help="mesh to train synthesizer on, usually unrepaired mesh")
parser.add_argument("input_mesh", type=str, default="mesh to use synthesizer on, usually reparation patch")
parser.add_argument("--no_cache", type=str2bool, default=False)

def synthesize(args):
  device = CUDA(0)
  # Generating Training Data
  gt_paths = [f'{const.DATA_ROOT}/{args.train_mesh}/{args.train_mesh}_level{i:02d}.obj' for i in range(6)]
  is_generated = all(list(os.path.isfile(gt_path) for gt_path in gt_paths))
  if (not is_generated) or args.no_cache:
    gen_args = options.GtOptions(tag='demo', mesh_name=args.train_mesh, template_name='sphere', num_levels=6)
    gt_gen = GroundTruthGenerator(gen_args, device)
  print("Finished generating training data with " + args.train_mesh, flush=True)

  # Training Synthesizer
  options_path = f'{const.PROJECT_ROOT}/checkpoints/{args.train_mesh}_demo/options.pkl'
  models_path = f'{const.PROJECT_ROOT}/checkpoints/{args.train_mesh}_demo/SingleMeshGenerator.pth'
  is_trained = os.path.isfile(options_path) and os.path.isfile(models_path)
  train_args = options.TrainOption(tag='demo', mesh_name=args.train_mesh, template_name='sphere', num_levels=6)
  if (not is_trained) or args.no_cache:
    trainer = Trainer(train_args, device)
    trainer.train()
  print("Finished training with " + args.train_mesh, flush=True)

  # Synthesizing Input
  m2m = Mesh2Mesh(train_args, CPU)
  mesh = mesh_utils.load_real_mesh(args.input_mesh, 0, True)
  out = m2m(mesh, 2, 5, 0)
  out.export(args.input_mesh + '_hi')
  print("Finished synthesizing input on " + args.input_mesh, flush=True)


if __name__ == '__main__':
  args, rest = parser.parse_known_args()
  synthesize(args)
