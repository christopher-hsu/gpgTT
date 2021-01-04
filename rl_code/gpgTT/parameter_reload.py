import gym
import gym_flock
import numpy as np
import pdb
import dgl
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from make_g import build_graph
import torch.optim as optim
import dgl
import dgl.function as fn
import math
import pdb, argparse

from torch.autograd import Variable
from torch.distributions import Categorical
import torch
import networkx as nx
import pdb
import matplotlib.pyplot as plt
from policy import Net
from make_g import build_graph
from utils import *
import time
import os 
import datetime 
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--map', type=str, default="emptyMed")
parser.add_argument('--nb_agents', type=int, default=4)
parser.add_argument('--nb_targets', type=int, default=4)
parser.add_argument('--seed', help='RNG seed', type=int, default=5)

args = parser.parse_args()


# first load old policy
policy = Net()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

## maEnvs
import envs
env = envs.make('setTracking-vGPG',
				'ma_target_tracking',
				render=bool(1),
				record=bool(0),
				directory='',
				ros=bool(0),
				map_name=args.map,
				num_agents=args.nb_agents,
				num_targets=args.nb_targets,
				is_training=True,
				)

filename = './logs/test_4a4t/model.pt'
policy.load_state_dict(torch.load(filename))


params_set = [
		{'nb_agents': 2, 'nb_targets': 2},
		{'nb_agents': 3, 'nb_targets': 3},
		{'nb_agents': 4, 'nb_targets': 4}
		]


def main():

	seed = args.seed
	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)

	test_episodes = 50
	plotting_rew = []
	

	for params in params_set:



		for episode in range(test_episodes):
			reward_over_eps = []
			state = env.reset(**params) # Reset environment and record the starting state
			g = build_graph(env)
			
			#need to do one dummy action through the policy to initialize it
			#because pytorch and its dynamic graph things. 
			#action = select_action(state,g,policy2)
			#set_weights(policy2,policy)

			#pdb.set_trace()
			done = False
			count = 0
			#for time in range(500):
			# while True:
			for time in range(200):
				#if episode%50==0:
				env.render()
				if episode==0 and count ==0:
					#pdb.set_trace()
					count += 1
					#time.sleep(15)
				#pdb.set_trace()
				g = build_graph(env)
				action = select_action(state,g,policy)

				action = action.numpy()
				# action = np.reshape(action,[-1])

				# Step through environment using chosen action
				# action = np.clip(action,-env.max_accel,env.max_accel)

				state, reward, done, info = env.step(action)

				# reward_over_eps.append(reward)
				reward_over_eps += reward['__all__']
				# Save reward
				policy.reward_episode.append(reward['__all__'])
				if done['__all__']:
					break

			# Used to determine when the environment is solved.
			#running_reward = (running_reward * 0.99) + (time * 0.01)


			if episode % 25 == 0:
				print('Episode {}\tAverage reward over episode: {:.2f}'.format(episode, reward_over_eps))

		

	# 	plotting_rew.append(np.mean(reward_over_eps))
	
	# np.savetxt('Test_Relative_Goal_Reaching_for_%d_agents_rs_rg.txt' %(env.n_agents), plotting_rew)
	# fig = plt.figure()
	# x = np.linspace(0,len(plotting_rew),len(plotting_rew))
	# plt.plot(x,plotting_rew)
	# plt.savefig('Test_Relative_Goal_Reaching_for_%d_agents_rs_rg.png' %(env.n_agents))
	# plt.show()


if __name__ == '__main__':
	main()