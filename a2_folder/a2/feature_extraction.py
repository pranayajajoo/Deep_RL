import numpy as np

def terrible_feature_extractor(obs, action):
    
    obs = obs[1:-1, 1:-1 ]
    obstacle, agent = component_extractor(obs)

    age_x, age_y, age_width, age_height = get_bounding_box(agent)
    obs_x, obs_y, obs_width, obs_height = get_bounding_box(obstacle) 
    
    dist_from_floor = obs_y - age_y
    dist_bet_o_a = obs_x - age_x - age_width
    jump_coords = obs_x - obs_height
    jump_here = (age_x + age_width) == jump_coords
    dist_from_jump = (obs_x - obs_height) - (age_x + age_width)

    distances = np.array([dist_from_floor, dist_bet_o_a, dist_from_jump]) / obs.shape[0]
    # jump_now = (age_x + age_width) == jump_point

    ac_0 = np.array([jump_here, not(jump_here)]) * (action == 0)
    ac_1 = np.array([jump_here, not(jump_here)]) * (action == 1)

    feat = np.concat((ac_0, ac_1, distances)) #np.concat(agent_pos, obstacle_pos)
    

    # import ipdb; ipdb.set_trace()
    return feat


def component_extractor(obs):
    obstacle = obs == 0.5
    floor = obs == 1
    floor_sum = np.sum(floor, axis=1)
    floor = np.where(floor_sum == obs.shape[1])[0]
    agent = obs == 1
    agent[floor,] = 0
    return obstacle, agent

def get_bounding_box(mask):
    """Finds the bounding box (x, y, width, height) of the largest connected component in the mask."""
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return None  # No object found
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return x_min, y_min, (x_max - x_min + 1), (y_max - y_min + 1)


def gen_masks(obs):
    
    obstacle_mask = obs == 0.5
    
    floor_mask = obs == 1
    floor_row = np.argmax(np.sum(floor_mask, 1))
    
    agent_mask = obs == 1
    agent_mask[floor_row, :] = False
    
    return obstacle_mask, agent_mask