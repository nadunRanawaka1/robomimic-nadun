

def demo_obs_to_obs_dict(demo_obs, ind):
    obs_dict = {}
    for o in demo_obs:
        obs_dict[o] = demo_obs[o][ind]
    return obs_dict