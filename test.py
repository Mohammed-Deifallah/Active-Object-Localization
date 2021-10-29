from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
import tqdm
from config import VOC2007_ROOT
from utils.dataset import read_voc_dataset

from utils.env import bbox_env
from config import *
from utils.tools import *

def get_loader():
    _,test_loader = read_voc_dataset(
                                path=f"{VOC2007_ROOT}/VOCtrainval_06-Nov-2007",
                                year='2007'
                              )
    return sort_class_extract([test_loader])

def create_env(img_loader, feature_extractor, fe_out_dim, stats_path, need_render=False):
    env = get_vectorized_env( bbox_env, img_loader, feature_extractor, fe_out_dim, 1, need_render )
    # check_env( unwrap_vec_env(env) )
    env = VecNormalize.load(stats_path, env)
    return env

def load_model(model_path):
    model = PPO.load(model_path)
    return model

def predict_image(env, predict_cb):
    obs = env.env_method('get_obs')
    gt_boxes = env.env_method('get_gt_boxes')
    done = False
    while not done:
        action = predict_cb(obs)
        obs, _, done, _ = env.step(action)
    return gt_boxes[0], env.env_method('get_final_box')[0]

def evaluate(class_name, env, predict_cb, n_images):
    env.reset()
    gt_boxes = []
    pred_boxes = []
    for _ in tqdm.tqdm(range(n_images)):
        gtruth, pred = predict_image(env, predict_cb)
        gt_boxes.append(gtruth)
        pred_boxes.append(pred)
    print("Computing recall and ap...")
    stats = eval_stats_at_threshold(pred_boxes, gt_boxes)
    print(f"Final result ({class_name}) : \n{stats}")

def predict_render_loop(env, predict_cb, n_steps):
    # e = unwrap_vec_env(env)
    env.training = False
    env.norm_reward = False
    obs = env.reset()
    for i in tqdm.tqdm(range(n_steps)):
        action = predict_cb(obs)
        obs, _, done, _ = env.step(action)
        env.render()

def test_class(img_loader, class_name, model_path, stats_path):
    n_images = len(img_loader[class_name])

    f_extr, f_extr_dim = get_feature_extractor(use_cuda)
    env = create_env(img_loader[class_name], f_extr, f_extr_dim, stats_path)

    model = load_model(model_path)

    model_predict_cb = lambda obs: model.predict(obs, deterministic=True)[0]
    random_predict_cb = lambda obs: [env.action_space.sample()]
    always_smaller_predict_cb = lambda obs: [6]

    predict_cb = model_predict_cb
    print(f"Evaluating model's performance for class '{class_name}'...")
    evaluate(class_name, env, predict_cb, n_images)

    env = create_env(img_loader[class_name], f_extr, f_extr_dim, stats_path, need_render=True)
    print(f"Rendering predictions for class '{class_name}'...")
    predict_render_loop(env, predict_cb, n_steps=100)


img_loader = get_loader()

for class_name, model_path, stats_path in zip(CLASSES, BEST_MODELS, STATS_PATHS):
    test_class(img_loader, class_name, model_path, stats_path)
