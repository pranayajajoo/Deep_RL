import numpy as np
import matplotlib.pyplot as plt


def terrible_feature_extractor(obs, action):
    
    obs = np.flip(obs, axis=0)
    obs = obs[1:-1, 1:-1 ]
    obstacle, agent = component_extractor(obs)

    age_x, age_y, age_width, age_height = get_bounding_box(agent)

    try:
        first_obstacle_mask = obstacle.copy()        
        first_obstacle_mask[:, 3*obs.shape[0]//4:] = 0
        # plt.imshow(first_obstacle_mask)
        # plt.show()
        # import ipdb; ipdb.set_trace()

        second_obstacle_mask = obstacle.copy()
        second_obstacle_mask[:, :3*obs.shape[0]//4] = 0
        # plt.imshow(second_obstacle_mask)
        # plt.show()
        # import ipdb; ipdb.set_trace()


        obs1_x, obs1_y, obs1_width, obs1_height = get_bounding_box(first_obstacle_mask)
        obs2_x, obs2_y, obs2_width, obs2_height = get_bounding_box(second_obstacle_mask) 

        jump_at_1 = obs1_x - obs1_height
        jump_at_2 = obs2_x - obs2_height
        # jump_here_1 = (age_x + age_width) == jump_at_1
        # jump_here_2 = (age_x + age_width) == jump_at_2

        dist_from_floor = obs1_y - age_y
        if age_x + age_width < obs1_x:
            dist_bet_o_a = obs1_x - age_x - age_width
            dist_from_jump = jump_at_1 - (age_x + age_width)
            jump_here = dist_from_jump == 0
        elif age_x + age_width < obs2_x:
            dist_bet_o_a = obs2_x - age_x - age_width
            dist_from_jump = jump_at_2 - (age_x + age_width)
            jump_here = dist_from_jump == 0
        else:
            dist_bet_o_a = 0
            dist_from_jump = 0
            jump_here = 0
        
        # if dist_bet_o1_a == 0:
        #     dist_from_jump_1 = 0
        # else:
        #     dist_from_jump_1 = (obs1_x - obs1_height) - (age_x + age_width)
        # if dist_bet_o2_a == 0:
        #     dist_from_jump_2 = 0
        # else:
        #     dist_from_jump_2 = (obs2_x - obs2_height) - (age_x + age_width)
        distances = np.array([dist_from_floor, dist_bet_o_a, dist_from_jump]) / obs.shape[0] # to prevent weights from blowing up
        # distances = np.array([dist_from_floor, dist_bet_o1_a, dist_from_jump_1]) / obs.shape[0] # to prevent weights from blowing up


        # only providing the jump point is sufficient for the agent to learn
        # incorporating actions
        # ac_0 = np.array([jump_here_1, not(jump_here_1)]) * (action == 0)
        # ac_1 = np.array([jump_here_1, not(jump_here_1)]) * (action == 1)   
        ac_0 = np.array([jump_here, not(jump_here)]) * (action == 0)
        ac_1 = np.array([jump_here, not(jump_here)]) * (action == 1)  

        # ac_1_0 = np.array([jump_here_1, not(jump_here_1)]) * (action == 0)
        # ac_1_1 = np.array([jump_here_1, not(jump_here_1)]) * (action == 1)
        # ac_2_0 = np.array([jump_here_2, not(jump_here_2)]) * (action == 0)
        # ac_2_1 = np.array([jump_here_2, not(jump_here_2)]) * (action == 1)   

    except:
        obs_x, obs_y, obs_width, obs_height = get_bounding_box(obstacle)
        jump_at = obs_x - obs_height
        jump_here = (age_x + age_width) == jump_at

        dist_from_floor = obs_y - age_y
        dist_bet_o_a = obs_x - age_x - age_width
        dist_from_jump = (obs_x - obs_height) - (age_x + age_width)
        distances = np.array([dist_from_floor, dist_bet_o_a, dist_from_jump]) / obs.shape[0] # normalizing to prevent weights from blowing up

        # only providing the jump point is sufficient for the agent to learn
        # incorporating actions
        ac_0 = np.array([jump_here, not(jump_here)]) * (action == 0)
        ac_1 = np.array([jump_here, not(jump_here)]) * (action == 1)   

    # dist_from_floor = age_y - obs_y
    # dist_bet_o_a = obs_x - (age_x + age_width)
    # jump_at = obs_x - obs_height
    # dist_from_jump = (obs_x - obs_height) - (age_x + age_width)
    # distances = np.array([dist_from_floor, dist_bet_o_a, dist_from_jump])

    # obs_x, obs_y, obs_width, obs_height = get_bounding_box(obstacle)
    # jump_at = obs_x - obs_height
    # jump_points = (age_x + age_width) == jump_at
        # jump_points.append(jump_now)

    # distances = np.array(distances) / obs.shape[0] # to prevent weights from blowing up

    
     

    feat = np.concat((ac_0, ac_1, distances))
    # feat = np.concatenate((ac_1_0, ac_1_1, ac_2_0, ac_2_1, distances))
    
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

# def get_all_bounding_boxes(mask):
#     """Finds bounding boxes of all connected components in the mask."""
#     labeled, num_features = ndimage.label(mask)
#     bounding_boxes = []

#     for i in range(1, num_features + 1):
#         component_mask = (labeled == i)
#         bounding_box = get_bounding_box(component_mask)
#         if bounding_box:
#             bounding_boxes.append(bounding_box)

#     return bounding_boxes
