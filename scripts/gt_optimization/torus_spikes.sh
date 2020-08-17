echo ground truth meshes will be saved under ./dataset/torus_spikes/
export PYTHONPATH=$PYTHONPATH:$PWD
python process_data/ground_truth_optimization.py --tag gt --mesh-name torus_spikes --template-name torus_spikes_template --num-levels 5