#!/usr/bin/env python
import os
import time
import optuna
import argparse
import plotext as plt
import traceback
from datetime import timedelta

import numpy as np
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

# from utils import clr, inv_clr

TUNE_PARAMS      = False
OPTUNA_STUDY_NAME='optuna_11'

N_LEARN_STEPS    = 1300500
N_TEST_STEPS     = 10100
BATCH_SIZE       = 2**10
STEPS_IN_EPISODE = 1000
N_CPU            = 8

def unwrap_vec_env( vec_env ):
    return vec_env.unwrapped.envs[0].unwrapped

def get_vectorized_env(episode_steps, n_envs):
    env_args = {
        'n_dof':n_dof,
        'grid_size':lfw.grid_size,
        'energy_callback':lfw.calc_energy,
        'opt_pose_callback':lfw.get_opt_pose,
        'steps_in_episode':episode_steps
    }
    env = make_vec_env(
                env_id=bbox_env,
                n_envs=n_envs,
                env_kwargs=env_args
            )
    env = VecNormalize(env)
    # env = VecCheckNan(env, raise_exception=True)
    return env

def fit_model( learn_env, n_learn_steps, batch_size, params={}, tb_log_dir=None ):
    model = PPO(
            'MultiInputPolicy',
            learn_env,
            verbose=1,
            device='cpu',
            batch_size = batch_size,
            **params,
            tensorboard_log=tb_log_dir
        )
    model.learn( total_timesteps=n_learn_steps, log_interval=1 )
    return model

def main():
    test_env = get_vectorized_env( STEPS_IN_EPISODE, n_envs=1 )
    check_env( unwrap_vec_env(test_env) )

    learn_env = get_vectorized_env(STEPS_IN_EPISODE, n_envs=N_CPU )
    start_time = time.time()
    params = {
        # "n_steps": BATCH_SIZE
    }
    model = fit_model(learn_env, N_LEARN_STEPS, BATCH_SIZE, params, tb_log_dir="./tensorboard_log/")

    f_name = os.path.join('models', time.strftime("ppo_%Y_%m_%d__%H_%M"))
    model.save(f_name+'.zip')
    venv = model.get_vec_normalize_env()
    if venv is not None:
        venv.save(f_name+'.pkl')

    mean_rew, std_rew = evaluate_policy( model, test_env, n_eval_episodes=10 )

    print(f"TOTAL TIME : {timedelta(seconds=time.time() - start_time)}" )
    print(f'Mean reward: {mean_rew:.3f} +/- {std_rew:.3f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--steps', type=int, default=N_LEARN_STEPS)
    args = parser.parse_args()
    N_LEARN_STEPS = args.steps
    main()
