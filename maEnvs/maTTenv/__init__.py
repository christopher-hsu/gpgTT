# from gym.envs.registration import register

# register(
#     id='maTracking-v0',
#     entry_point='maTTenv.env:maTrackingEnv0',
# )
# register(
#     id='setTracking-v0',
#     entry_point='maTTenv.env:setTrackingEnv0',
# )


from utilities.ma_time_limit import maTimeLimit

def make(env_name, render=False, figID=0, record=False, ros=False, directory='',
                    T_steps=None, num_agents=2, num_targets=1, **kwargs):
    """
    env_name : str
        name of an environment. (e.g. 'Cartpole-v0')
    type : str
        type of an environment. One of ['atari', 'classic_control',
        'classic_mdp','target_tracking']
    """
    if T_steps is None:
        T_steps = 200

    if env_name == 'maTracking-v0':
        from maTTenv.env.maTracking_v0 import maTrackingEnv0
        env0 = maTrackingEnv0(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'maTracking-v1':
        from maTTenv.env.maTracking_v1 import maTrackingEnv1
        env0 = maTrackingEnv1(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'maTracking-v2':
        from maTTenv.env.maTracking_v2 import maTrackingEnv2
        env0 = maTrackingEnv2(num_agents=num_agents, num_targets=num_targets, **kwargs)

    elif env_name == 'setTracking-v0':
        from maTTenv.env.setTracking_v0 import setTrackingEnv0
        env0 = setTrackingEnv0(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-v1':
        from maTTenv.env.setTracking_v1 import setTrackingEnv1
        env0 = setTrackingEnv1(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-v0-noShuffle':
        from maTTenv.env.setTracking_v0_noShuffle import setTrackingEnv0noShuffle
        env0 = setTrackingEnv0noShuffle(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-vGreedy':
        from maTTenv.env.setTracking_vGreedy import setTrackingEnvGreedy
        env0 = setTrackingEnvGreedy(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-vkGreedy':
        from maTTenv.env.setTracking_vkGreedy import setTrackingEnvkGreedy
        env0 = setTrackingEnvkGreedy(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-veryGreedy':
        from maTTenv.env.setTracking_veryGreedy import setTrackingEnvVeryGreedy
        env0 = setTrackingEnvVeryGreedy(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-vGru':
        from maTTenv.env.setTracking_vGru import setTrackingEnvGru
        env0 = setTrackingEnvGru(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-vGPG':
        from maTTenv.env.setTracking_vGPG import setTrackingEnvGPG
        env0 = setTrackingEnvGPG(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'setTracking-vGPGdisc':
        from maTTenv.env.setTracking_vGPGdisc import setTrackingEnvGPGdisc
        env0 = setTrackingEnvGPGdisc(num_agents=num_agents, num_targets=num_targets, **kwargs)
    elif env_name == 'advTracking-v0':
        from maTTenv.env.advTracking_v0 import advTrackingEnv0
        env0 = advTrackingEnv0(num_agents=num_agents, num_targets=num_targets, **kwargs)
    else:
        raise ValueError('No such environment exists.')

    env = maTimeLimit(env0, max_episode_steps=T_steps)

    if ros:
        from ttenv.ros_wrapper import Ros
        env = Ros(env)
    if render:
        from maTTenv.display_wrapper import Display2D
        env = Display2D(env, figID=figID)
    if record:
        from maTTenv.display_wrapper import Video2D
        env = Video2D(env, dirname = directory)

    return env