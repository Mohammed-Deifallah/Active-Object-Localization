#!/usr/bin/env python
import os
import time
import argparse
from datetime import timedelta

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from torch.autograd import Variable

np.seterr(under='ignore')

import torch
torch.set_num_threads(1)
torch.autograd.set_detect_anomaly(True)

from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from utils.env import bbox_env
from utils.tools import *
from utils.models import *
from utils.dataset import read_voc_dataset

# from utils import clr, inv_clr

N_ENV      = 10
N_ROLLOUTS = 10
N_ROLLOUT_STEPS = 100
N_SAVE_ROOT = os.path.join('models', time.strftime(f'%Y%b%d_%H_%M'))

def unwrap_vec_env( vec_env ):
    return vec_env.unwrapped.envs[0].unwrapped

def get_vectorized_env(image_loader, feature_extractor, fe_out_dim, n_envs):
    args = {
        'image_loader': image_loader,
        'feature_extractor': feature_extractor,
        'feature_extractor_dim': fe_out_dim
    }
    env = make_vec_env(
                env_id=bbox_env,
                n_envs=n_envs,
                env_kwargs=args
            )
    env = VecNormalize(env)
    # env = VecCheckNan(env, raise_exception=True)
    return env

def fit_model( class_name, train_env, params={}, tb_log_dir=None, eval_env=None ):
    model = PPO(
            'MlpPolicy',
            train_env,
            n_epochs = 5,
            n_steps  = N_ROLLOUT_STEPS,
            verbose  = 1,
            # device = 'cpu',
            batch_size = N_ROLLOUT_STEPS * train_env.num_envs,
            **params,
            tensorboard_log=tb_log_dir
        )
    eval_callback = EvalCallback(
                            eval_env,
                            best_model_save_path=f'{N_SAVE_ROOT}/{class_name}',
                            log_path='./logs/',
                            eval_freq=N_ROLLOUT_STEPS,
                            deterministic=True,
                            render=False
                    )
    model.learn(
        callback = eval_callback,
        total_timesteps = N_ROLLOUTS * N_ROLLOUT_STEPS * train_env.num_envs
    )
    return model

def save_model(class_name, model):
    f_name = os.path.join('models', time.strftime(f'{class_name}_%Y%b%d_%H_%M'))
    model.save(f_name+'.zip')
    venv = model.get_vec_normalize_env()
    if venv is not None:
        venv.save(f_name+'.pkl')

def load_model(env, model_path, stats_path):
    model = PPO.load(model_path)
    env = VecNormalize.load(stats_path, env)
    return model, env

def train_class(class_name, image_loader, feature_extractor, fe_out_dim):
    eval_env = get_vectorized_env( image_loader, feature_extractor, fe_out_dim, n_envs=1 )
    check_env( unwrap_vec_env(eval_env) )
    train_env = get_vectorized_env( image_loader, feature_extractor, fe_out_dim, n_envs=N_ENV )
    params = {
        # "n_steps": BATCH_SIZE
    }
    model = fit_model(
                class_name,
                train_env,
                params,
                tb_log_dir="./tensorboard_log/",
                eval_env=eval_env
            )
    save_model( class_name, model )
    del train_env
    return model, eval_env

def validate_class(class_name, model, eval_env):
    mean_rew, std_rew = evaluate_policy( model, eval_env, n_eval_episodes=10 )
    print(f'Class {class_name}: mean reward = {mean_rew:.3f} +/- {std_rew:.3f}')

def get_feature_extractor(use_cuda):
    feature_extractor = FeatureExtractor(network='vgg16')
    feature_extractor.eval()
    if use_cuda:
        feature_extractor = feature_extractor.cuda()
    feature_extractor_dim = 25088
    return feature_extractor, feature_extractor_dim

def get_features(feature_extractor, image, dtype=FloatTensor):
    global transform
    #image = transform(image)
    image = image.view(1,*image.shape)
    image = Variable(image).type(dtype)
    if use_cuda:
        image = image.cuda()
    feature = feature_extractor(image)
    #print("Feature shape : "+str(feature.shape))
    return feature.data

def main():

    train_loader2007_train, train_loader2007_val = \
        read_voc_dataset(path="../data/VOCtrainval_06-Nov-2007" ,year='2007')
    dsets_per_class_train = sort_class_extract([train_loader2007_train])
    dsets_per_class_val   = sort_class_extract([train_loader2007_val])

    fe, fe_out_dim = get_feature_extractor(use_cuda)
    feature_extractor = lambda img: get_features(fe,img)

    start_time = time.time()

    for i in range(len(classes)):
        class_name = classes[i]
        print(f"Training class : {class_name} ...")
        model, eval_env = train_class(
                                class_name,
                                dsets_per_class_train[class_name],
                                feature_extractor,
                                fe_out_dim
                                )
        validate_class( class_name, model, eval_env )
        del model
        del eval_env
        torch.cuda.empty_cache()

    print(f"TOTAL TIME : {timedelta(seconds=time.time() - start_time)}" )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--rollouts', type=int, default=N_ROLLOUTS)
    args = parser.parse_args()
    N_ROLLOUTS = args.steps
    main()
