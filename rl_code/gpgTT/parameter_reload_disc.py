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
import matplotlib
import matplotlib.pyplot as plt
# from policy import Net
from policy import DiscreteNet
from make_g import build_graph
from utils import *
import time
import os, pickle
import datetime 
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--map', type=str, default="emptyMed")
parser.add_argument('--nb_agents', type=int, default=4)
parser.add_argument('--nb_targets', type=int, default=4)
parser.add_argument('--seed', help='RNG seed', type=int, default=5)
parser.add_argument('--log_dir', type=str, default='./logs/test_4a4t/')
args = parser.parse_args()


# first load old policy
policy = DiscreteNet()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

## maEnvs
import envs
env = envs.make('setTracking-vGPGdisc',
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

filename = args.log_dir+'model.pt'
policy.load_state_dict(torch.load(filename))


params_set = [
		# {'nb_agents': 2, 'nb_targets': 2},
		# {'nb_agents': 3, 'nb_targets': 3},
		# {'nb_agents': 4, 'nb_targets': 4},
		# {'nb_agents': 8, 'nb_targets': 8},
		{'nb_agents': 20, 'nb_targets': 20}
		]


def main():

	seed = args.seed
	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)

	test_episodes = 50
	plotting_rew = []
	total_mean_rew = []

	for params in params_set:
		eps_rewards = []


		for episode in range(test_episodes):

			reward_over_eps = 0
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
				# env.render()
				# if episode==0 and count ==0:
					#pdb.set_trace()
					# count += 1
					#time.sleep(15)
				#pdb.set_trace()
				g = build_graph(env)
				action = select_discrete_action(state,g,policy)

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

			eps_rewards.append(reward_over_eps)
			
			if episode % 50 == 0:
				print('Episode {}\tAverage reward over episode: {:.2f}'.format(episode, reward_over_eps))

		mean_rew_over_eps = np.mean(eps_rewards)
		total_mean_rew.append(mean_rew_over_eps)
		
		eval_dir = os.path.join(args.log_dir, 'eval_seed%d_'%(seed)+args.map)
		if not os.path.exists(eval_dir):
			os.makedirs(eval_dir)
		matplotlib.use('Agg')
		f0, ax0 = plt.subplots()
		_ = ax0.plot(eps_rewards, '.')
		_ = ax0.set_title('setTracking-vGPGdisc')
		_ = ax0.set_xlabel('episode number')
		_ = ax0.set_ylabel('mean reward')
		_ = ax0.axhline(y=mean_rew_over_eps, color='r', linestyle='-', label='mean over episodes: %.2f'%(mean_rew_over_eps))
		_ = ax0.legend()
		_ = ax0.grid()
		_ = f0.savefig(os.path.join(eval_dir, "%da%dt_%d_eval_"%(env.nb_agents, env.nb_targets, test_episodes)
												+".png"))
		plt.close()
		pickle.dump(eps_rewards, open(os.path.join(eval_dir,"%da%dt_%d_eval_"%(env.nb_agents, env.nb_targets, test_episodes))
																+".pkl", 'wb'))

	#Plot over all example episode sets
	f1, ax1 = plt.subplots()
	_ = ax1.plot(total_mean_rew, '.')
	_ = ax1.set_title('setTracking-vGPGdisc')
	_ = ax1.set_xlabel('example episode set number')
	_ = ax1.set_ylabel('mean return over episodes')
	_ = ax1.grid()
	_ = f1.savefig(os.path.join(eval_dir,'all_%d_eval'%(test_episodes)+'.png'))
	plt.close()        
	pickle.dump(total_mean_rew, open(os.path.join(eval_dir,'all_%d_eval'%(test_episodes))+'%da%dt'%(args.nb_agents,args.nb_targets)+'.pkl', 'wb'))


if __name__ == '__main__':
	main()