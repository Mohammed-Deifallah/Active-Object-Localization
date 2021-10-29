from random import random
import gym
import torch
from gym import spaces
import numpy as np

from config import *
from utils.tools import *
import torchvision.ops as ops

# from utils.models import *

class bbox_env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            image_loader,
            feature_extractor,
            feature_extractor_dim,
            need_render = False,
            nu=3.0,
            alpha=0.2,
            threshold=0.5 ):
        super().__init__()

        self.image_loader = image_loader
        self.img_keys = list(image_loader.keys())
        self.key_idx = -1

        self.feature_extractor = feature_extractor

        self.need_render = need_render

        self.nu    = nu # Reward of Trigger
        self.alpha = alpha # €[0, 1]  Scaling factor
        self.threshold = threshold

        xmin = 0.0
        xmax = 224.0
        ymin = 0.0
        ymax = 224.0

        self.orig_box = [xmin, xmax, ymin, ymax]
        self.cur_box  = self.orig_box

        self.reset()

        self.n_actions = 9
        self.action_space = spaces.Discrete(self.n_actions)

        state_dim = self.n_actions**2 + feature_extractor_dim

        self.observation_space = spaces.Box(
            low  = np.array( [-np.inf]*state_dim ),
            high = np.array( [np.inf]*state_dim ),
            dtype= np.float32
        )
        self.max_episode_steps = 20
        self._max_episode_steps = 20
        self.name = 'BBox_env_v0'

    def rewrap(self, coord, mi=0, ma=224):
        return min(max(coord,mi), ma)

    def calculate_position_box(self, actions, x_min=0, x_max=224, y_min=0, y_max=224):

        for r in actions:
            alpha_h = self.alpha * (  y_max - y_min )
            alpha_w = self.alpha * (  x_max - x_min )

            if r == 1: # Right
                x_min += alpha_w
                x_max += alpha_w
            if r == 2: # Left
                x_min -= alpha_w
                x_max -= alpha_w
            if r == 3: # Up
                y_min -= alpha_h
                y_max -= alpha_h
            if r == 4: # Down
                y_min += alpha_h
                y_max += alpha_h
            if r == 5: # Bigger
                y_min -= alpha_h
                y_max += alpha_h
                x_min -= alpha_w
                x_max += alpha_w
            if r == 6: # Smaller
                y_min += alpha_h
                y_max -= alpha_h
                x_min += alpha_w
                x_max -= alpha_w
            if r == 7: # Fatter
                y_min += alpha_h
                y_max -= alpha_h
            if r == 8: # Taller
                x_min += alpha_w
                x_max -= alpha_w
        x_min = self.rewrap(x_min,0,223)
        x_max = self.rewrap(x_max+1,1,224)
        y_min = self.rewrap(y_min,0,223)
        y_max = self.rewrap(y_max+1,1,224)
        return [x_min, x_max, y_min, y_max]

    def intersection_over_union(self, box1, box2):
        x11, x21, y11, y21 = box1
        x12, x22, y12, y22 = box2
        box1 = Tensor([x11,y11,x21,y21]).view(1,-1)
        box2 = Tensor([x12,y12,x22,y22]).view(1,-1)
        return ops.box_iou( box1, box2 ).item()

    def get_max_iou_box(self, gt_boxes, cur_box):
        max_iou = False
        max_gt = []
        for gt in gt_boxes:
            iou = self.intersection_over_union(cur_box, gt)
            if max_iou == False or max_iou < iou:
                max_iou = iou
                max_gt = gt
        return max_gt

    def compute_trigger_reward(self, box, gt_box):
        res = self.intersection_over_union(box, gt_box)
        self.iou_dif = 0.0
        self.iou = res
        if res>=self.threshold:
            return self.nu
        return -1*self.nu

    def compute_reward(self, curr_state, prev_state, gt_box):
        prev_iou = self.intersection_over_union(prev_state, gt_box)
        curr_iou = self.intersection_over_union(curr_state, gt_box)
        iou_dif = curr_iou-prev_iou
        self.iou = curr_iou
        self.iou_dif = iou_dif
        return -2.0 if iou_dif <= 0 else 1.0

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
        history_flatten = self.actions_history.view(1,-1)
        state = torch.cat((image_feature, history_flatten), 1)
        return state.squeeze().cpu().numpy()

    def update_history(self, action):
        action_vector = get_tensor( torch.zeros, 9 )
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
        self.cur_step += 1
        self.all_actions.append(action)

        if action == 0:
            self.state = self.compose_state(self.img)
            self.cur_box = self.calculate_position_box( self.all_actions )
            closest_gt_box = self.get_max_iou_box( self.gt_boxes, self.cur_box )
            self.reward = self.compute_trigger_reward( self.cur_box,  closest_gt_box )
            self.done = True
        else:
            self.actions_history = self.update_history( action )
            new_box = self.calculate_position_box( self.all_actions )

            new_img = self.orig_img[:, int(new_box[2]):int(new_box[3]), int(new_box[0]):int(new_box[1])]
            self.img = transform(new_img)


            self.state = self.compose_state( self.img )
            closest_gt_box = self.get_max_iou_box( self.gt_boxes, new_box )
            self.reward = self.compute_reward( new_box, self.cur_box, closest_gt_box )
            self.cur_box = new_box

        if self.cur_step >= self._max_episode_steps:
            self.done = True

        if self.need_render:
            self.render()

        return self.state, self.reward, self.done, {}

    def reset(self):
        self.final_box = self.cur_box
        self.done = False
        self.cur_step = 0
        self.all_actions = []
        self.iou = 0.
        self.iou_dif = 0.
        self.reward = None
        self.key_idx = (self.key_idx+1) % len(self.img_keys)
        self.key = self.img_keys[ self.key_idx ]
        # print(self.key)
        self.img, gt_boxes = extract(self.key, self.image_loader)
        self.orig_img = self.img.clone()
        self.gt_boxes = gt_boxes
        self.cur_box = self.orig_box
        self.actions_history = get_tensor( torch.ones, (9,9) )
        self.state = self.compose_state(self.orig_img)
        return self.state

    def render(self, mode='human'):
        file_prefix=self.key[:6]+'_'
        text = [f'iou:{self.iou:.2f}', f'diff:{self.iou_dif:.2f}', f'reward:{self.reward}']
        show_new_bdbox(
            self.orig_img,
            self.cur_box, count=self.cur_step,
            infos=text,
            save_path=MEDIA_ROOT, prefix=file_prefix )
        # if self.done:
        #     make_movie(
        #         in_dir='media', in_prefix=file_prefix, out_dir=MEDIA_ROOT, total_frames=self.cur_step )
    def get_obs(self):
        return self.state
    def get_gt_boxes(self):
        return self.gt_boxes
    def get_final_box(self):
        return self.final_box
