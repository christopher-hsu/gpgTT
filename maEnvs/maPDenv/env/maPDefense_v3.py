import os, copy, pdb
import numpy as np
from numpy import linalg as LA
from gym import spaces, logger
from maPDenv.maps import map_utils
import maPDenv.util as util 
from maPDenv.agent_models import *
from maPDenv.belief_tracker import *
from maPDenv.metadata import METADATA_pd_v2
from maPDenv.policies import *
from maPDenv.env.maPDefense_Base import maPDefenseBase

"""
Perimeter Defense Environments for Reinforcement Learning.
[Variables]

d: radial coordinate of a belief target in the learner frame
alpha : angular coordinate of a belief target in the learner frame
ddot : radial velocity of a belief target in the learner frame
alphadot : angular velocity of a belief target in the learner frame
Sigma : Covariance of a belief target
o_d : linear distance to the closet obstacle point
o_alpha : angular distance to the closet obstacle point
p_d : linear distance to the perimeter
p_alpha : angular distance to the perimeter

[Environment Description]
Varying number of agents, varying number of sprialing targets
Defend the perimeter!
No obstacles
Infinite sensing range with scaling noise

maPDefenseEnv2 : SE2 Target model with UKF belief tracker
    obs state: [d, alpha, ddot, alphadot, logdet(Sigma), observed, spot, intruder, p_d, p_alpha] *nb_targets
    obs state: [d, alpha, ddot, alphadot, logdet(Sigma), observed, o_d, o_alpha] *nb_targets
            where nb_targets and nb_agents vary between a range
            num_targets describes the upperbound on possible number of targets in env
            num_agents describes the upperbound on possible number of agents in env
    Target : SE2 model [x,y,theta,v,w] + a control policy u=[v,w]
    Belief Target : UKF for SE2Vel model [x,y,theta,v,w]

"""

class maPDefenseEnv3(maPDefenseBase):

    def __init__(self, num_agents=1, num_targets=2, map_name='empty', 
                        is_training=True, known_noise=True, **kwargs):
        super().__init__(num_agents=num_agents, num_targets=num_targets,
                        map_name=map_name, is_training=is_training)

        self.id = 'maPDefense-v3'
        self.nb_agents = num_agents #only for init, will change with reset()
        self.nb_targets = num_targets #only for init, will change with reset()
        self.agent_dim = 3
        self.target_dim = 5
        self.target_init_vel = np.array(METADATA['target_init_vel'])
        self.perimeter_radius = METADATA['perimeter_radius']
        # LIMIT
        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin, [-np.pi, -METADATA['target_vel_limit'], -np.pi])),
                                np.concatenate((self.MAP.mapmax, [np.pi, METADATA['target_vel_limit'], np.pi]))]
        rel_vel_limit = METADATA['target_vel_limit'] + METADATA['action_v'][0] # Maximum relative speed
        self.limit['state'] = np.array([[0.0, -np.pi, -rel_vel_limit, -10*np.pi, -50.0, 0.0, 0.0, -1.0, 0.0, -np.pi ],
                                        [600.0, np.pi, rel_vel_limit, 10*np.pi,  50.0, 1.0, 1.0, 0.0, self.sensor_r, np.pi]])
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

        self.target_noise_cov = np.zeros((self.target_dim, self.target_dim))
        for i in range(3):
            self.target_noise_cov[i,i] = METADATA['const_q'] * self.sampling_period**3/3
        self.target_noise_cov[3:, 3:] = METADATA['const_q'] * \
                    np.array([[self.sampling_period, self.sampling_period**2/2],
                             [self.sampling_period**2/2, self.sampling_period]])
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * \
                                  self.sampling_period * np.eye(self.target_dim)

        # Build a robot 
        self.setup_agents()
        # Build a target
        self.setup_targets()
        self.setup_belief_targets()

    def setup_agents(self):
        self.agents = [AgentSE2(agent_id = 'agent-' + str(i), 
                        dim=self.agent_dim, sampling_period=self.sampling_period, 
                        limit=self.limit['agent'], 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_agents)]

    def setup_targets(self):
        self.targets = [AgentSE2(agent_id = 'target-' + str(i),
                        dim=self.target_dim, sampling_period=self.sampling_period, 
                        limit=self.limit['target'],
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                        policy=SpiralPolicy(self.sampling_period, self.MAP.origin, 
                                            METADATA['spiral_min'], METADATA['spiral_max']))
                        for i in range(self.num_targets)]


    def setup_belief_targets(self):
        self.belief_targets = [UKFbelief(agent_id = 'target-' + str(i),
                        dim=self.target_dim, limit=self.limit['target'], 
                        fx=SE2DynamicsVel,
                        W=self.target_noise_cov, 
                        obs_noise_func=self.observation_noise,
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_targets)]


    def get_reward(self, obstacles_pt=None, observed=None, is_training=True):
        return self.reward_fun(observed, self.origin_init_pos, 
                            self.perimeter_radius, is_training)

    def reward_fun(self, observed, goal_origin, goal_radius, 
                    is_training=True, c_mean=0.1):
        """ Return a reward for targets that enter the goal radius or observed
        -1 for entering goal radius
        """
        # tracking reward
        detcov = [LA.det(b_target.cov) for b_target in self.belief_targets[:self.nb_targets]]
        r_detcov_mean = -np.mean(np.log(detcov))
        # reward = c_mean * r_detcov_mean
        reward = c_mean * (r_detcov_mean - 10.57)

        # perimeter defense reward
        intruder = observed.astype(float)
        target_states = [target.state[:3] for target in self.targets[:self.nb_targets]]
        global_states = util.global_relative_measure(target_states, goal_origin)
        intruder[global_states[:,0] < goal_radius] = -50

        #if captured or entered goal reset target pose
        for ii, rew in enumerate(intruder):
            if rew != 0:
                self.reset_target_pose(target_id=ii)

        intruder[intruder>0] = 0
        tot_intruder = np.sum(intruder)
        reward += tot_intruder
        reward += 0.5 #for ep len

        done = False
        if tot_intruder < 0:
            done = True

        info_dict = {'mean_nlogdetcov': r_detcov_mean, 
                     'num_intruders': tot_intruder, 'intruders': intruder}

        return reward, done, info_dict

    def reset(self,**kwargs):
        """
        Random initialization a number of agents and targets at the reset of the env epsiode.
        Agents are given random positions in the map, targets are given random positions near a random agent.
        Return an observation state dict with agent ids (keys) that refer to their observation
        """
        self.rng = np.random.default_rng()
        try: 
            # self.nb_agents = kwargs['nb_agents']
            self.nb_targets = kwargs['nb_targets']
        except:
            # self.nb_agents = np.random.random_integers(1, self.num_agents)
            self.nb_targets = np.random.random_integers(1, self.num_targets)
        obs_dict = {}
        init_pose = self.get_init_pose(**kwargs)
        # Initialize agents
        for ii in range(self.nb_agents):
            self.agents[ii].reset(init_pose['agents'][ii])
            obs_dict[self.agents[ii].agent_id] = []

        # Initialize targets and beliefs
        for nn in range(self.nb_targets):
            self.belief_targets[nn].reset(
                        init_state=np.concatenate((init_pose['belief_targets'][nn], np.zeros(2))),
                        init_cov=self.target_init_cov)
            t_init = np.concatenate((init_pose['targets'][nn], [self.target_init_vel[0], 0.0]))
            self.targets[nn].reset(t_init)
        # For nb agents calculate belief of targets assigned
        for jj in range(self.nb_targets):
            for kk in range(self.nb_agents):
                r, alpha = util.relative_distance_polar(self.belief_targets[jj].state[:2],
                                            xy_base=self.agents[kk].state[:2], 
                                            theta_base=self.agents[kk].state[2])
                r_perim, a_perim = util.relative_distance_polar(self.origin_init_pos[:2],
                                            xy_base=self.agents[kk].state[:2], 
                                            theta_base=self.agents[kk].state[2])
                logdetcov = np.log(LA.det(self.belief_targets[jj].cov))
                obs_dict[self.agents[kk].agent_id].append([r, alpha, 0.0, 0.0, logdetcov, 
                                                           0.0, 0.0, 0.0, r_perim, a_perim])
        for agent_id in obs_dict:
            obs_dict[agent_id] = np.asarray(obs_dict[agent_id])
        return obs_dict

    def step(self, action_dict):
        obs_dict = {}
        reward_dict = {}
        done_dict = {'__all__':False}
        info_dict = {}
        all_observations = np.zeros(self.nb_targets, dtype=bool)

        # Targets move (t -> t+1)
        for n in range(self.nb_targets):
            self.targets[n].update() 
            # self.belief_targets[n].predict(np.array([np.random.random(),
            #                                         np.pi*np.random.random()-0.5*np.pi]))
        # Agents move (t -> t+1) and observe the targets
        for ii, agent_id in enumerate(action_dict):
            obs_dict[self.agents[ii].agent_id] = []
            reward_dict[self.agents[ii].agent_id] = []
            done_dict[self.agents[ii].agent_id] = []

            action_vw = self.action_map[action_dict[agent_id]]

            # Locations of targets and agents in order to maintain a margin between them
            margin_pos = [t.state[:2] for t in self.targets[:self.nb_targets]]
            for p, ids in enumerate(action_dict):
                if agent_id != ids:
                    margin_pos.append(np.array(self.agents[p].state[:2]))
            _ = self.agents[ii].update(action_vw, margin_pos)
            
            # Target and map observations
            observed = np.zeros(self.nb_targets, dtype=bool)
            # obstacles_pt = map_utils.get_closest_obstacle(self.MAP, self.agents[ii].state)
            # if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

            # Update beliefs of targets using UKF
            for jj in range(self.nb_targets):
                # Observe
                obs, z_t, spot = self.observation(self.targets[jj], self.agents[ii])
                observed[jj] = obs

                #if spotted than target has also been observed
                # self.belief_targets[jj].update(spot, z_t, self.agents[ii].state)
                # if spot:
                self.belief_targets[jj].update(spot, z_t, self.agents[ii].state,
                                            np.array([np.random.random(),
                                            np.pi*np.random.random()-0.5*np.pi]))

                r_b, alpha_b = util.relative_distance_polar(self.belief_targets[jj].state[:2],
                                        xy_base=self.agents[ii].state[:2], 
                                        theta_base=self.agents[ii].state[-1])
                r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                                        self.belief_targets[jj].state[:2],
                                        self.belief_targets[jj].state[2:],
                                        self.agents[ii].state[:2], self.agents[ii].state[-1],
                                        action_vw[0], action_vw[1])
                r_perim, a_perim = util.relative_distance_polar(self.origin_init_pos[:2],
                                            xy_base=self.agents[ii].state[:2], 
                                            theta_base=self.agents[ii].state[2])
                obs_dict[agent_id].append([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                        np.log(LA.det(self.belief_targets[jj].cov)), 
                                        float(obs), float(spot), 0.0, r_perim, a_perim])
                                        # float(obs + spot), 0.0, obstacles_pt[0], obstacles_pt[1]])
            obs_dict[agent_id] = np.asarray(obs_dict[agent_id])
            all_observations = np.logical_or(all_observations, observed)

        # Get all rewards after all agents and targets move (t -> t+1)
        reward, done, info_dict = self.get_reward(obstacles_pt, all_observations, self.is_training)
        reward_dict['__all__'], done_dict['__all__'] = reward, done
        for kk, agent_id in enumerate(obs_dict):
            obs_dict[agent_id][:,7] = info_dict['intruders']
            # self.rng.shuffle(obs_dict[agent_id])
        return obs_dict, reward_dict, done_dict, info_dict

    def observation(self, target, agent):
        r, alpha = util.relative_distance_polar(target.state[:2],
                                            xy_base=agent.state[:2], 
                                            theta_base=agent.state[2])    
        # observed is a bool for capturing targets with short range sensor
        observed = (r <= self.sensor_r) \
                    & (abs(alpha) <= self.fov/2/180*np.pi) \
                    & (not(map_utils.is_blocked(self.MAP, agent.state, target.state)))
        # spotted is a bool for scouting targets with long range sensor
        spotted = (r <= self.sensor_r_long) \
                    & (abs(alpha) <= self.fov_long/2/180*np.pi) \
                    & (not(map_utils.is_blocked(self.MAP, agent.state, target.state)))
        z = None

        if spotted:
            z = np.array([r, alpha])
            z += self.np_random.multivariate_normal(np.zeros(2,), self.observation_noise(z))
        
        if observed:
            z = np.array([r, alpha])
            # z += np.random.multivariate_normal(np.zeros(2,), self.observation_noise(z))
            # z += self.np_random.multivariate_normal(np.zeros(2,), self.observation_noise(z))
        '''For some reason, self.np_random is needed only here instead of np.random in order for the 
        RNG seed to work, if used in the gen_rand_pose functions RNG seed will NOT work '''

        return observed, z, spotted

    def observation_noise(self, z, c=0.1):
        obs_noise_cov = (c * z[0])**2 * np.array([[self.sensor_r_sd * self.sensor_r_sd, 0.0],
                                        [0.0, self.sensor_b_sd * self.sensor_b_sd]])
        # obs_noise_cov = np.array([[self.sensor_r_sd * self.sensor_r_sd, 0.0],
                                        # [0.0, self.sensor_b_sd * self.sensor_b_sd]])
        return obs_noise_cov

    def reset_target_pose(self, target_id,
                lin_dist_range_target=(METADATA['target_init_dist_min'], METADATA['target_init_dist_max']),
                ang_dist_range_target=(-np.pi, np.pi),
                lin_dist_range_belief=(METADATA['init_belief_distance_min'], METADATA['init_belief_distance_max']),
                ang_dist_range_belief=(-np.pi, np.pi),
                blocked=False):
        """if captured or entered goal reset target pose and belief
        """
        is_target_valid = False
        while(not is_target_valid):
            is_target_valid, init_pose_target = self.gen_rand_pose(
                self.origin_init_pos[:2], self.origin_init_pos[2],
                lin_dist_range_target[0], lin_dist_range_target[1],
                ang_dist_range_target[0], ang_dist_range_target[1])

        is_belief_valid, init_pose_belief = False, np.zeros((2,))
        while((not is_belief_valid) and is_target_valid):
            is_belief_valid, init_pose_belief = self.gen_rand_pose(
                init_pose_target[:2], init_pose_target[2],
                lin_dist_range_belief[0], lin_dist_range_belief[1],
                ang_dist_range_belief[0], ang_dist_range_belief[1])

        self.belief_targets[target_id].reset(
                    init_state=np.concatenate((init_pose_belief,np.zeros(2))),
                    init_cov=self.target_init_cov)
        t_init = np.concatenate((init_pose_target,[self.target_init_vel[0], 0.0]))
        self.targets[target_id].reset(t_init)