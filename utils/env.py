import gym
import torch
from gym import spaces
import numpy as np

from config import *
# from utils.models import *


# from utils import clr, inv_clr

class bbox_env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__( self, class_name, get_image_cb, threshold, nu, alpha, feature_extractor):
        super().__init__()

        self.class_name   = class_name
        self.get_image_cb = get_image_cb

        self.nu    = nu # Reward of Trigger
        self.alpha = alpha # €[0, 1]  Scaling factor

        self.threshold = threshold

        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()

        self.reset()

        xmin = 0.0
        xmax = 224.0
        ymin = 0.0
        ymax = 224.0

        self.orig_coord = [xmin, xmax, ymin, ymax]
        self.actu_coord = self.orig_coord

        self.n_actions = 9
        self.action_space = spaces.Discrete(self.n_actions)

        state_dim = 81 + feature_extractor.dim

        self.observation_space = spaces.Box(
            low  = np.array( [-np.inf]*state_dim ),
            high = np.array( [np.inf]*state_dim ),
            dtype= np.float32
        )

        # self.observation_space = spaces.Dict({
        #     'energy'     : energy_space,
        #     'prev_energy': energy_space,
        #     'reward'     : reward_space,
        #     'prev_reward': reward_space,
        #     'action'     : self.action_space,
        #     'prev_action': self.action_space,
        #     'pose'       : pose_space,
        #     'prev_pose'  : pose_space
        # })

    def rewrap(self, coord):
        return min(max(coord,0), 224)

    def calculate_position_box(self, actions, xmin=0, xmax=224, ymin=0, ymax=224):
        alpha_h = self.alpha * (  ymax - ymin )
        alpha_w = self.alpha * (  xmax - xmin )
        real_x_min, real_x_max, real_y_min, real_y_max = 0, 224, 0, 224

        for r in actions:
            if r == 1: # Right
                real_x_min += alpha_w
                real_x_max += alpha_w
            if r == 2: # Left
                real_x_min -= alpha_w
                real_x_max -= alpha_w
            if r == 3: # Up
                real_y_min -= alpha_h
                real_y_max -= alpha_h
            if r == 4: # Down
                real_y_min += alpha_h
                real_y_max += alpha_h
            if r == 5: # Bigger
                real_y_min -= alpha_h
                real_y_max += alpha_h
                real_x_min -= alpha_w
                real_x_max += alpha_w
            if r == 6: # Smaller
                real_y_min += alpha_h
                real_y_max -= alpha_h
                real_x_min += alpha_w
                real_x_max -= alpha_w
            if r == 7: # Fatter
                real_y_min += alpha_h
                real_y_max -= alpha_h
            if r == 8: # Taller
                real_x_min += alpha_w
                real_x_max -= alpha_w
        real_x_min, real_x_max, real_y_min, real_y_max = self.rewrap(real_x_min), self.rewrap(real_x_max), self.rewrap(real_y_min), self.rewrap(real_y_max)
        return [real_x_min, real_x_max, real_y_min, real_y_max]

    def intersection_over_union(self, box1, box2):
        x11, x21, y11, y21 = box1
        x12, x22, y12, y22 = box2

        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

    def get_max_bdbox(self, gt_boxes, actu_coord ):
        max_iou = False
        max_gt = []
        for gt in gt_boxes:
            iou = self.intersection_over_union(actu_coord, gt)
            if max_iou == False or max_iou < iou:
                max_iou = iou
                max_gt = gt
        return max_gt

    def compute_trigger_reward(self, box, gt_box):
        res = self.intersection_over_union(box, gt_box)
        if res>=self.threshold:
            return self.nu
        return -1*self.nu

    def get_features(self, image, dtype=torch.FloatTensor):
        global transform
        #image = transform(image)
        feature = self.feature_extractor(image)
        #print("Feature shape : "+str(feature.shape))
        return feature.data

    def compose_state(self, image, dtype=torch.FloatTensor):
        image_feature = self.get_features(image, dtype)
        image_feature = image_feature.view(1,-1)
        #print("image feature : "+str(image_feature.shape))
        history_flatten = self.actions_history.view(1,-1).type(dtype)
        state = torch.cat((image_feature, history_flatten), 1)
        return state

    def update_history(self, action):
        action_vector = torch.zeros(9)
        action_vector[action] = 1
        size_history_vector = len(torch.nonzero(self.actions_history))
        if size_history_vector < 9:
            self.actions_history[size_history_vector][action] = 1
        else:
            for i in range(8,0,-1):
                self.actions_history[i][:] = self.actions_history[i-1][:]
            self.actions_history[0][:] = action_vector[:]
        return self.actions_history

    def step(self, action):
        self.step += 1
        self.all_actions.append(action)

        if action == 0:
            state = None
            new_coord = self.calculate_position_box(self.all_actions)
            closest_gt = self.get_max_bdbox(  self.gt_boxes, new_coord )
            reward = self.compute_trigger_reward(new_coord,  closest_gt)
            done = True
        else:
            self.actions_history = self.update_history(action)
            new_coord = self.calculate_position_box(self.all_actions)

            new_image = self.orig_img[:, int(new_coord[2]):int(new_coord[3]), int(new_coord[0]):int(new_coord[1])]
            try:
                new_image = transform(new_image)
            except ValueError:
                done = True

            state = self.compose_state(new_image)
            closest_gt = self.get_max_bdbox( self.gt_boxes, new_coord )
            reward = self.compute_reward(new_coord, self.actu_coord, closest_gt)
            self.actu_coord = new_coord

        if self.step == 20:
            done = True

        return state, reward, done, {}

    def reset(self):
        img, gt_boxes = self.get_image_cb(self.class)
        self.orig_img = img.clone()
        self.gt_boxes = gt_boxes
        self.actions_history = np.ones((9,9))
        self.state = self.compose_state(self.orig_img)
        return self.state

    def render(self, mode='human', force=False):
        # if self.energy < self.best_energy:
        if force or self.need_print():
            if self.cur_step==0 :
                print('**********************RESET*********************')
            print(f'Step        : {self.cur_step}')
            print(f'Action      : {self.action}')
            print(f'Pose        : {self.pose}')
            print(f'Reward      : {clr(self.reward)}')
            print(f'Cum reward  : {clr(self.cum_reward)}')
            print(f'Obs         : {self.obs}')
            print(f'Start energy: {self.start_nrg:.5f}')
            print(f'Energy      : {self.nrg:.5f}')
            print(f'Tot nrg diff: {clr(self.get_energy_diff())} %')
            print( '-------------')
            # self.best_nrg = self.nrg

    def need_print(self):
        return np.random.random() < 3./self.steps_in_episode