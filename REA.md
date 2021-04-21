# gpgTT
Graph Policy Gradient for Target Tracking
by Christopher Hsu

Adapted for the multi-agent target tracking environment

From,

Graph Policy Gradients for Large Scale Robot Control
In CoRL 2019, Oral Paper [Paper][Video][Blog Post][CoRL Talk]
Arbaaz Khan, Ekaterina Tolstaya, Alejandro Ribeiro, Vijay Kumar
GRASP Lab, University of Pennsylvania



This repository contains the code for our paper Graph Policy Gradients for Large Scale Robot Control. The idea is to train a large number of agents by exploiting local structure using Graph Convolutional Networks. Training code as well as inference code is included in this implementation.

Installation and Usage
First install the formation flying environment from inside gym_formation/. Instructions provided in the enclosed readme file.

Install deep graph library (https://www.dgl.ai/) and compatible version of PyTorch.

Running the code:
cd rl_code
#To train GPG from scratch. (See additional point below)
python3 main.py 
#Inference for a larger swarm (loads model from ./logs)
python3 parameter_reload.py 
A simple arrowhead formation is specified in constr_formation_flying.py (inside gym_formation/gym_flock/envs/) To increase the number of agents simply change the parameter self.n_agents in constr_formation_flying.py, line 39.

Make sure to change the number of agents to a small number if training from scratch (Ideally < 10)
