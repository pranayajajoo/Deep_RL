import numpy as np
import matplotlib.pyplot as plt

def get_bounding_box_agent(mask):

    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return x_min.item(), y_min.item(), (x_max - x_min + 1).item(), (y_max - y_min + 1).item()


def gen_masks(obs):
    
    obstacle_mask = obs == 0.5
    
    floor_mask = obs == 1
    floor_row = np.argmax(np.sum(floor_mask, 1))
    
    agent_mask = obs == 1
    agent_mask[floor_row, :] = False
    
    return obstacle_mask, agent_mask

def terrible_feature_extractor(obs, action):
    
    # preprocess the observation
    obs = np.flip(obs, axis=0)
    obs = obs[1:-1, 1:-1 ]

    width, height = obs.shape

    obstacle_mask_primary, agent_mask = gen_masks(obs)
    
    try:
        
        # Tries to extract 2 obstacles
        # HACK - assumes that the second one lies on the extreme right end of the screen - hence the 9/10

        obstacle_mask_secondary = obstacle_mask_primary.copy()
        obstacle_mask_secondary[:, :9*width//10] = 0 # only second block

        obstacle_mask_primary[:, 9*width//10:] = 0 # only first block
        
        b2_x, b2_y, b2_w, b2_h = get_bounding_box_agent(obstacle_mask_secondary)
        # This above line throws an error if there is no more than 1 obstacle
        
        config = 3
        
    except:
        
        config = 1.5

    # (x, y, width, height)
    a_x, a_y, a_w, a_h = get_bounding_box_agent(agent_mask)
    b_x, b_y, b_w, b_h = get_bounding_box_agent(obstacle_mask_primary)    
    
    floor_level = b_y + b_h
    agent_bottom = a_y + a_h
    
    gap = floor_level - agent_bottom
    
    dist_from_right = width - (a_x + a_w)
    dist_from_block1 = b_x - (a_x + a_w)

    jump_point1 = dist_from_block1 == b_h
    distance_to_jump_point1 = dist_from_block1 - b_h
    
    if config == 3:
        
        dist_from_block2 = b2_x - (a_x + a_w)
        
        jump_point2 = dist_from_block2 == b2_h
        
        distance_to_jump_point2 = dist_from_block2 - b2_h
    
        common_feat = np.array([dist_from_right, distance_to_jump_point1, distance_to_jump_point2, gap, 1]) / width

        feat_0 = np.array([jump_point1, not(jump_point1), jump_point2, not(jump_point2)]) * (action == 0)
        feat_1 = np.array([jump_point1, not(jump_point1), jump_point2, not(jump_point2)]) * (action == 1)

    
    else:

        common_feat = np.array([dist_from_right, distance_to_jump_point1, gap, 1]) / width

        feat_0 = np.array([jump_point1, not(jump_point1)]) * (action == 0)
        feat_1 = np.array([jump_point1, not(jump_point1)]) * (action == 1)

    return np.concat((common_feat, feat_0, feat_1))
