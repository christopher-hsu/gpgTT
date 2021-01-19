import os, copy, pdb
import numpy as np
from numpy import linalg as LA
from gym import spaces, logger
from maTTenv.maps import map_utils
import maTTenv.util as util 
from maTTenv.agent_models import *
from maTTenv.belief_tracker import KFbelief
from maTTenv.metadata import METADATA
from maTTenv.env.maTracking_Base import maTrackingBase
## gpg
from sklearn.neighbors import NearestNeighbors

"""
Target Tracking Environments for Reinforcement Learning.
[Variables]

d: radial coordinate of a belief target in the learner frame
alpha : angular coordinate of a belief target in the learner frame
ddot : radial velocity of a belief target in the learner frame
alphadot : angular velocity of a belief target in the learner frame
Sigma : Covariance of a belief target
o_d : linear distance to the closet obstacle point
o_alpha : angular distance to the closet obstacle point

[Environment Description]
made for Graph Policy Gradients with discete actions
Varying number of agents, varying number of randomly moving targets
No obstacles
greedy target assignment
continuous actions

setTrackingEnv0 : Double Integrator Target model with KF belief tracker
    obs state: [d, alpha, ddot, alphadot, logdet(Sigma), observed, o_d, o_alpha] *nb_targets
            where nb_targets and nb_agents vary between a range
            num_targets describes the upperbound on possible number of targets in env
            num_agents describes the upperbound on possible number of agents in env
    Target : Double Integrator model, [x,y,xdot,ydot]
    Belief Target : KF, Double Integrator model

"""

class setTrackingEnvGPGdisc(maTrackingBase):

    def __init__(self, num_agents=1, num_targets=2, map_name='empty', 
                        is_training=True, known_noise=True, **kwargs):
        super().__init__(num_agents=num_agents, num_targets=num_targets,
                        map_name=map_name, is_training=is_training)

        self.id = 'setTracking-vGPGdisc'
        self.nb_agents = num_agents #only for init, will change with reset()
        self.nb_targets = num_targets #only for init, will change with reset()
        self.agent_dim = 3
        self.target_dim = 4
        self.target_init_vel = METADATA['target_init_vel']*np.ones((2,))
        # LIMIT
        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,[-METADATA['target_vel_limit'], -METADATA['target_vel_limit']])),
                                np.concatenate((self.MAP.mapmax, [METADATA['target_vel_limit'], METADATA['target_vel_limit']]))]
        rel_vel_limit = METADATA['target_vel_limit'] + METADATA['action_v'][0] # Maximum relative speed
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -rel_vel_limit, -10*np.pi, -50.0, 0.0], [0.0, -np.pi ])),
                               np.concatenate(([600.0, np.pi, rel_vel_limit, 10*np.pi,  50.0, 2.0], [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

        # a_vel_lim = 2.0
        # a_ang_lim = np.pi/2
        # self.limit['action'] = [np.array([0.0, -a_ang_lim]), np.array([a_vel_lim, a_ang_lim])]
        # self.action_space = spaces.Box(self.limit['action'][0], self.limit['action'][1], dtype=np.float32)

        self.targetA = np.concatenate((np.concatenate((np.eye(2), self.sampling_period*np.eye(2)), axis=1), 
                                        [[0,0,1,0],[0,0,0,1]]))
        self.target_noise_cov = METADATA['const_q'] * np.concatenate((
                        np.concatenate((self.sampling_period**3/3*np.eye(2), self.sampling_period**2/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * np.concatenate((
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))

        #gpg params
        self.n_neighbors = 1
        # self.mean_pooling = False  #normalize the adjacency matrix by the number of neighbors

        # Build a robot 
        self.setup_agents()
        # Build a target
        self.setup_targets()
        self.setup_belief_targets()
        # Use custom reward
        self.get_reward()

    def setup_agents(self):
        self.agents = [AgentSE2(agent_id = 'agent-' + str(i), 
                        dim=self.agent_dim, sampling_period=self.sampling_period, 
                        limit=self.limit['agent'], 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_agents)]

    def setup_targets(self):
        self.targets = [AgentDoubleInt2D(agent_id = 'target-' + str(i),
                        dim=self.target_dim, sampling_period=self.sampling_period, 
                        limit=self.limit['target'],
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                        A=self.targetA, W=self.target_true_noise_sd) 
                        for i in range(self.num_targets)]

    def setup_belief_targets(self):
        self.belief_targets = [KFbelief(agent_id = 'target-' + str(i),
                        dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
                        W=self.target_noise_cov, obs_noise_func=self.observation_noise, 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_targets)]

    def get_reward(self, obstacles_pt=None, observed=None, is_training=True):
        return reward_fun(self.nb_targets, self.belief_targets, is_training)

    def reset(self,**kwargs):
        """
        Random initialization a number of agents and targets at the reset of the env epsiode.
        Agents are given random positions in the map, targets are given random positions near a random agent.
        Return an observation state dict with agent ids (keys) that refer to their observation
        """
        try: 
            self.nb_agents = kwargs['nb_agents']
            self.nb_targets = kwargs['nb_targets']
        except:
            self.nb_agents = 4#np.random.random_integers(2, self.num_agents)
            self.nb_targets = 4#np.random.random_integers(1, self.num_targets)
        obs_dict = {}
        state = []
        self.greedy_dict = {}   #dict to see which target is assigned to which agent
        self.graph_x = []
        init_pose = self.get_init_pose(**kwargs)
        # Initialize agents
        for ii in range(self.nb_agents):
            self.agents[ii].reset(init_pose['agents'][ii])
            obs_dict[self.agents[ii].agent_id] = []
            self.greedy_dict[self.agents[ii].agent_id] = []

        # Initialize targets and beliefs
        for nn in range(self.nb_targets):
            self.belief_targets[nn].reset(
                        init_state=np.concatenate((init_pose['belief_targets'][nn][:2], np.zeros(2))),
                        init_cov=self.target_init_cov)
            self.targets[nn].reset(np.concatenate((init_pose['targets'][nn][:2], self.target_init_vel)))
        # For nb agents calculate belief of targets assigned
        for jj in range(self.nb_targets):
            for kk in range(self.nb_agents):
                r, alpha = util.relative_distance_polar(self.belief_targets[jj].state[:2],
                                            xy_base=self.agents[kk].state[:2], 
                                            theta_base=self.agents[kk].state[2])
                logdetcov = np.log(LA.det(self.belief_targets[jj].cov))
                obs_dict[self.agents[kk].agent_id].append([r, alpha, 0.0, 0.0, logdetcov, 
                                                           0.0, self.sensor_r, np.pi])
        # Greedily assign agents to closest target in order, all targets assigned if agents > targets
        mask = np.ones(self.nb_targets,bool)
        if self.nb_targets > self.nb_agents:
            oracle=1
        else:
            oracle=0
        for agent_id in obs_dict:
            obs_dict[agent_id] = np.asarray(obs_dict[agent_id])
            if np.sum(mask) != np.maximum(0,self.nb_targets-self.nb_agents+oracle):
                idx = np.flatnonzero(mask)
                self.greedy_dict[agent_id] = idx[np.argmin(obs_dict[agent_id][:,0][mask])]
                obs_dict[agent_id] = obs_dict[agent_id][None,self.greedy_dict[agent_id]]
                mask[self.greedy_dict[agent_id]] = False
            # self.graph_x.append(obs_dict[agent_id][:,:2])
            self.graph_x.append(self.belief_targets[self.greedy_dict[agent_id]].state[:2])
            state.append(obs_dict[agent_id])
        self.graph_x = np.squeeze(np.asarray(self.graph_x))
        state = np.squeeze(np.asarray(state))
        # return obs_dict
        return state

    def step(self, action):
        obs_dict = {}
        state = []
        reward_dict = {}
        done_dict = {'__all__':False}
        info_dict = {}
        action_dict = {}

        # clip actions into vel and angvel limits and put into dict
        # action = np.clip(action, self.limit['action'][0], self.limit['action'][1])
        for ii in range(self.nb_agents):
            action_vw = self.action_map[action[ii][0]]
            action_dict[self.agents[ii].agent_id] = action_vw

        # Targets move (t -> t+1)
        for n in range(self.nb_targets):
            self.targets[n].update() 
            self.belief_targets[n].predict() # Belief state at t+1
        # Agents move (t -> t+1) and observe the targets
        for ii, agent_id in enumerate(action_dict):
            obs_dict[self.agents[ii].agent_id] = []
            reward_dict[self.agents[ii].agent_id] = []
            done_dict[self.agents[ii].agent_id] = []

            # action_vw = self.action_map[action_dict[agent_id]]

            # Locations of all targets and agents in order to maintain a margin between them
            margin_pos = [t.state[:2] for t in self.targets[:self.nb_targets]]
            for p, ids in enumerate(action_dict):
                if agent_id != ids:
                    margin_pos.append(np.array(self.agents[p].state[:2]))
            _ = self.agents[ii].update(action_dict[agent_id], margin_pos)
            
            # Target and map observations
            observed = np.zeros(self.nb_targets, dtype=bool)
            # obstacles_pt = map_utils.get_closest_obstacle(self.MAP, self.agents[ii].state)
            # if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

            # Update beliefs of targets
            for jj in range(self.nb_targets):
                # Observe
                obs, z_t = self.observation(self.targets[jj], self.agents[ii])
                observed[jj] = obs
                if obs: # if observed, update the target belief.
                    self.belief_targets[jj].update(z_t, self.agents[ii].state)

                r_b, alpha_b = util.relative_distance_polar(self.belief_targets[jj].state[:2],
                                        xy_base=self.agents[ii].state[:2], 
                                        theta_base=self.agents[ii].state[-1])
                r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                                        self.belief_targets[jj].state[:2],
                                        self.belief_targets[jj].state[2:],
                                        self.agents[ii].state[:2], self.agents[ii].state[-1],
                                        action_dict[agent_id][0], action_dict[agent_id][1])
                obs_dict[agent_id].append([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                        np.log(LA.det(self.belief_targets[jj].cov)), 
                                        float(obs), obstacles_pt[0], obstacles_pt[1]])
            obs_dict[agent_id] = np.asarray(obs_dict[agent_id])

        # Assign target
        for agent_id in obs_dict:
            obs_dict[agent_id] = np.asarray(obs_dict[agent_id])
            obs_dict[agent_id] = obs_dict[agent_id][None,self.greedy_dict[agent_id]]
            state.append(obs_dict[agent_id])
        state = np.squeeze(np.asarray(state))
        # Get all rewards after all agents and targets move (t -> t+1)
        reward, done, mean_nlogdetcov = self.get_reward(obstacles_pt, observed, self.is_training)
        reward_dict['__all__'], done_dict['__all__'], info_dict['mean_nlogdetcov'] = reward, done, mean_nlogdetcov
        # return obs_dict, reward_dict, done_dict, info_dict
        return state, reward_dict, done_dict, info_dict

    ## Graph Policy Gradient related functions

    def dist2_mat(self, x):

        x_loc = np.reshape(x[:, 0:2], (self.nb_agents,2,1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        return a_net

    def get_connectivity(self, x):

        # if self.degree == 0:
            # a_net = self.dist2_mat(x)
            # a_net = (a_net < self.comm_radius2).astype(float)
        # else:
        neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        # neigh.fit(x[:,2:4])
        neigh.fit(x)
        a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())

        # if self.mean_pooling:
        #     # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
        #     n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.nb_agents,1)) # TODO or axis=0? Is the mean in the correct direction?
        #     n_neighbors[n_neighbors == 0] = 1
        #     a_net = a_net / n_neighbors

        return a_net




def reward_fun(nb_targets, belief_targets, is_training=True, c_mean=0.1):
    detcov = [LA.det(b_target.cov) for b_target in belief_targets[:nb_targets]]
    r_detcov_mean = - np.mean(np.log(detcov))# - np.std(np.log(detcov))
    reward = c_mean * r_detcov_mean

    mean_nlogdetcov = None
    if not(is_training):
        logdetcov = [np.log(LA.det(b_target.cov)) for b_target in belief_targets[:nb_targets]]
        mean_nlogdetcov = -np.mean(logdetcov)
    return reward, False, mean_nlogdetcov