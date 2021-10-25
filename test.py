from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from config import VOC2007_ROOT
from utils.dataset import read_voc_dataset

from utils.env import bbox_env
from config import *
from utils.tools import *

class_name = 'cat'
BEST_MODEL = 'models/2021Oct24_23_47/cat/best_model.zip'
STATS_PATH = 'models/cat_2021Oct25_06_08.pkl'

def get_loader():
    _,test_loader = read_voc_dataset(
                                path=f"{VOC2007_ROOT}/VOCtrainval_06-Nov-2007",
                                year='2007'
                              )
    return sort_class_extract([test_loader])

def create_env(img_loader, feature_extractor, fe_out_dim, stats_path):
    env = get_vectorized_env( bbox_env, img_loader, feature_extractor, fe_out_dim, 1, True )
    check_env( unwrap_vec_env(env) )
    env = VecNormalize.load(stats_path, env)
    return env

def load_model(model_path):
    model = PPO.load(model_path)
    return model

def predict_image(env, model):
    e = unwrap_vec_env(env)
    obs = env.reset()
    gt_boxes = e.gt_boxes
    done = False
    while not done:
        action,_ = model.predict( obs, deterministic=True )
        obs, _, done, _ = env.step(action)
    return gt_boxes, e.final_box

def evaluate(env, model, n_images):
    gt_boxes = []
    pred_boxes = []
    print("Predicting boxes...")
    for _ in range(n_images):
        gt_bxs, pred_box = predict_image(env, model)
        gt_boxes.append(gt_bxs)
        pred_boxes.append(pred_box)
    print("Computing recall and ap...")
    stats = eval_stats_at_threshold(pred_boxes, gt_boxes)
    print("Final result : \n"+str(stats))

def predict_loop(env, predict_cb, n_steps):
    # e = unwrap_vec_env(env)
    env.training = False
    env.norm_reward = False
    obs = env.reset()
    for i in range(n_steps):
        # action = env.action_space.sample()
        action, _ = predict_cb(obs)
        obs, _, done, _ = env.step(action)
        # env.render()

img_loader = get_loader()
n_images = len(img_loader)

f_extr, f_extr_dim = get_feature_extractor(use_cuda)
env = create_env(img_loader[class_name], f_extr, f_extr_dim, STATS_PATH)

model = load_model(BEST_MODEL)
evaluate(env, model, n_images)

# predict_cb = lambda obs: model.predict(obs, deterministic=True)
# predict_loop(env, predict_cb, n_steps=100)