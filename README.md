

### Deep Geometric Texture Synthesis in PyTorch
<img src='https://drive.google.com/uc?export=view&id=1t6WvvyvyZD3_A3XE_mumPBxS1RuJGyue' align="right" width=300>
<b>SIGGRAPH 2020 <a href="https://arxiv.org/abs/2007.00074" target="_blank">[Paper]</a> <a href="https://ranahanocka.github.io/geometric-textures/" target="_blank">[Project Page]</a></b>
<br><br>
Deep Geometric Texture Synthesis is an approach for learning the local geometric textures present within a 3D mesh model. This can be used to learn the unknown 3D geometric texture statistics from a single 3D model, and then synthesizes them on different 3D models.
This repository contains the code for: 
<br><br>
(1) creating multi-scale training data 
<br><br>
(2) training a series of multi-scale generators
<br><br>
(3) synthesizing the learned geometric textures on unseen models

#### Installation
- Clone this repo `git clone https://github.com/amirhertz/dgts`.
- Install via conda environment `conda env create -f environment.yml` (creates an environment called dgts)
#
### Training Demo

#### Download Example Data
First get the multi-scaled training inputs already prepared by running
```bash
bash ./scripts/get_train_data.sh
```

#### Running Training
The example scripts can be found in `scripts/train`. If using conda env first activate env e.g. `conda activate dgts`, then from the root directory:
```bash
bash ./scripts/train/virus_ball.sh
```
will train on the spikey-ball from the paper. There is also a demo script for the "sphere rail" and the lizard.
#
### Inference Demo

#### Get Trained Weights & Some Demo Data
```bash
bash ./scripts/inference/get_pretrained_data.sh
```
Note that if you already ran the training demo from above, this will overwrite some your training snapshots. 

#### Unconditional & Coarse Mesh Generation 
This will generate unconditional & conditioned on coarse mesh generative results (Fig. 5 from the paper):
```bash
bash ./scripts/inference/anky_generate.sh
```

#### Progressively Add Geometric Texture
This will generate a series of progressive textures. The target mesh will progressively gain textures, starting from a low-level, generator, up to a finer resolution generator. This results in a series of <i>animated</i> textures.
```bash
bash ./scripts/inference/sphere_rail_animate.sh
```
#
### Create Training Data Demo

#### Download Example Data
First get some example 3D meshes with geometric texture
```bash
bash ./scripts/gt_optimization/get_demo_data.sh
```

#### Running Optimization
The example scripts can be found in `scripts/gt_optimization`. For example, run the following from the root directory
```bash
bash ./scripts/gt_optimization/covid.sh
```
which will generate the coronavirus from the paper.
#
### Citation
If you find this code useful, please consider citing our paper
```
@article{Hertz2020deep,
  title = {Deep Geometric Texture Synthesis},
  author = {Hertz, Amir and Hanocka, Rana and Giryes, Raja and Cohen-Or, Daniel},
  year = {2020},
  issue_date = {July 2020}, 
  publisher = {Association for Computing Machinery}, 
  volume = {39}, 
  number = {4}, 
  issn = {0730-0301},
  url = {https://doi.org/10.1145/3386569.3392471},
  doi = {10.1145/3386569.3392471},
  articleno = {108},
  journal = {ACM Trans. Graph.} 
}
```
#
### Questions / Issues
If you have questions or issues running this code, please open an issue.
