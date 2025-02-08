import numpy as np


def terrible_feature_extractor(obs, action):
    
    obs = np.flip(obs, axis=0)
    obs = obs[1:-1, 1:-1 ]
    obstacle, agent = component_extractor(obs)

    age_x, age_y, age_width, age_height = get_coordinates(agent)

    try:
        first_obstacle_mask = obstacle.copy()        
        first_obstacle_mask[:, 3*obs.shape[0]//4:] = 0

        second_obstacle_mask = obstacle.copy()
        second_obstacle_mask[:, :3*obs.shape[0]//4] = 0

        obs1_x, obs1_y, obs1_width, obs1_height = get_coordinates(first_obstacle_mask)
        obs2_x, obs2_y, obs2_width, obs2_height = get_coordinates(second_obstacle_mask) 

        jump_at_1 = obs1_x - obs1_height
        jump_at_2 = obs2_x - obs2_height

        dist_from_floor = obs1_y - age_y
        if age_x + age_width < obs1_x:
            dist_bet_o_a = obs1_x - age_x - age_width
            dist_from_jump = jump_at_1 - (age_x + age_width)
            jump_here = dist_from_jump == 0
            obs1, obs2 = True, False
        else:
            dist_bet_o_a = obs2_x - age_x - age_width
            dist_from_jump = jump_at_2 - (age_x + age_width)
            jump_here = dist_from_jump == 0
            obs1, obs2 = False, True

        distances = np.array([dist_from_floor, dist_bet_o_a, dist_from_jump]) / obs.shape[0] # to prevent weights from blowing up

        # incorporating actions
        ac_0 = np.array([jump_here, not(jump_here)]) * (action == 0)
        ac_1 = np.array([jump_here, not(jump_here)]) * (action == 1)

        # obstacle flag to identify which obstacle is next (first or second)
        obs = np.array([obs1, obs2])  

    except:
        obs_x, obs_y, obs_width, obs_height = get_bounding_box(obstacle)
        jump_at = obs_x - obs_height
        jump_here = (age_x + age_width) == jump_at

        dist_from_floor = obs_y - age_y
        dist_bet_o_a = obs_x - age_x - age_width
        dist_from_jump = (obs_x - obs_height) - (age_x + age_width)
        distances = np.array([dist_from_floor, dist_bet_o_a, dist_from_jump]) / obs.shape[0] # normalizing to prevent weights from blowing up

        # incorporating actions
        ac_0 = np.array([jump_here, not(jump_here)]) * (action == 0)
        ac_1 = np.array([jump_here, not(jump_here)]) * (action == 1)

        # since there is only one obstacle
        obs = np.array([True])

    features = np.concat((ac_0, ac_1, distances, obs))

    return features

def component_extractor(obs):
    obstacle = obs == 0.5
    floor = obs == 1
    floor_sum = np.sum(floor, axis=1)
    floor = np.where(floor_sum == obs.shape[1])[0]
    agent = obs == 1
    agent[floor,] = 0
    return obstacle, agent

def get_coordinates(mask):
    position = np.column_stack(np.where(mask > 0))
    if position.size == 0:
        return None  # no object found
    y_min, x_min = position.min(axis=0)
    y_max, x_max = position.max(axis=0)
    return x_min, y_min, (x_max - x_min + 1), (y_max - y_min + 1)

