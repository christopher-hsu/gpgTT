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
import pdb, argparse, pickle

from torch.autograd import Variable
from torch.distributions import Categorical
import torch
import networkx as nx
import pdb
import matplotlib.pyplot as plt
from policy import Net
from policy import DiscreteNet
#from linear_policy import Net
from make_g import build_graph
from utils import *

import os
import datetime
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--map', type=str, default="emptyMed")
parser.add_argument('--nb_agents', type=int, default=4)
parser.add_argument('--nb_targets', type=int, default=4)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--discrete', type=int, default=0)

args = parser.parse_args()


torch_threads=1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.set_num_threads(torch_threads)

if args.discrete:
	policy = DiscreteNet()
	name = '_disc'
	env_name = 'setTracking-vGPGdisc'
else:
	policy = Net()#.to(device)
	name = '_cont'
	env_name = 'setTracking-vGPG'

## maEnvs
import envs
env = envs.make(env_name,
				'ma_target_tracking',
				render=bool(0),
				record=bool(0),
				directory='',
				ros=bool(0),
				map_name=args.map,
				num_agents=args.nb_agents,
				num_targets=args.nb_targets,
				is_training=True,
				)


optimizer = optim.Adam(policy.parameters(), lr=1e-3)
# env = gym.make('FormationFlying-v3')

if not os.path.exists('./logs'):
	os.makedirs('./logs')

filename = str(datetime.datetime.now().strftime("%m%d%H%M"))+name+str('_%da%dt_seed%d'%(env.nb_agents,env.nb_targets,args.seed))
savedir = str('./logs/%s'%filename)

if not os.path.exists(savedir):
	os.makedirs(savedir)

torch.save(policy.state_dict(), savedir+'/model.pt')

writer = SummaryWriter(savedir)


def main(episodes):

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	running_reward = 10
	plotting_rew = []
	reward_over_time = []
	lr_iter = 0

	for episode in range(episodes):
		reward_over_eps = 0
		state = env.reset() # Reset environment and record the starting state
		g = build_graph(env)
		done = False

		for time in range(200):

			#if episode%50==0:
			# env.render()
			#g = build_graph(env)
			if args.discrete:
				action = select_discrete_action(state,g,policy)
			else:
				action = select_action(state,g,policy)
			action = action.numpy()
			# action = np.reshape(action,[-1])

			# Step through environment using chosen action
			# action = np.clip(action,-env.max_accel,env.max_accel)

			state, reward, done, info = env.step(action)

			# reward_over_eps.append(reward['__all__'])
			reward_over_eps += reward['__all__']

			# Save reward
			policy.reward_episode.append(reward['__all__'])

			lr_iter += 1

			if done['__all__']:
				break


		# Used to determine when the environment is solved.
		# running_reward = (running_reward * 0.99) + (time * 0.01)
		reward_over_time.append(reward_over_eps)

		update_policy(policy,optimizer, lr_iter)

		if episode % 500 == 0:
			print('Episode {}\tAverage reward over episode: {:.2f}'.format(episode, reward_over_eps))

		if episode % 5000 == 0 :
			torch.save(policy.state_dict(), savedir+'/model.pt')


		# plotting_rew.append(np.mean(reward_over_eps))

		# Tensorboard logger

		writer.add_scalar('rewards', reward_over_eps, episode)
		writer.add_scalar('loss', policy.loss_history[-1], episode)
		writer.add_scalar('lr', policy.lr_history[-1],episode)


	pickle.dump(reward_over_time, open(savedir+'/rewards.pkl', 'wb'))

	#pdb.set_trace()
	# np.savetxt('Relative_Goal_Reaching_for_%d_agents_rs_rg.txt' %(env.n_agents), plotting_rew)
	# fig = plt.figure()
	# x = np.linspace(0,len(plotting_rew),len(plotting_rew))
	# plt.plot(x,plotting_rew)
	# plt.savefig('Relative_Goal_Reaching_for_%d_agents_rs_rg.png' %(env.n_agents))
	# plt.show()

	#pdb.set_trace()

episodes = 1
main(episodes)











#pdb.set_trace()
