
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from config import *
import pandas as pd
import random
import tqdm as tq
import imageio

from torch.autograd import Variable
from utils.models import FeatureExtractor
from stable_baselines3.common.env_util import make_vec_env



def sort_class_extract(datasets):
    datasets_per_class = {}
    for j in CLASSES:
        datasets_per_class[j] = {}

    for dataset in datasets:
        for i in tq.tqdm(dataset):
            # if random.random() > 0.8: return datasets_per_class
            img, target = i
            obj = target['annotation']['object']
            if isinstance(obj, list):
                classe = target['annotation']['object'][0]["name"]
            else:
                classe = target['annotation']['object']["name"]
            filename = target['annotation']['filename']

            org = {}
            for j in CLASSES:
                org[j] = []
                org[j].append(img)

            if isinstance(obj, list):
                for j in range(len(obj)):
                    classe = obj[j]["name"]
                    if classe in CLASSES:
                        org[classe].append([obj[j]["bndbox"], target['annotation']['size']])
            else:
                if classe in CLASSES:
                    org[classe].append([obj["bndbox"], target['annotation']['size']])
            for j in CLASSES:
                if len(org[j]) > 1:
                    try:
                        datasets_per_class[j][filename].append(org[j])
                    except KeyError:
                        datasets_per_class[j][filename] = []
                        datasets_per_class[j][filename].append(org[j])
    return datasets_per_class


def show_new_bdbox(image, labels, color='r', count=0, infos=[], save_path='', prefix=''):
    xmin, xmax, ymin, ymax = labels[0],labels[1],labels[2],labels[3]
    fig,ax = plt.subplots(1)
    ax.imshow(image.transpose(0, 2).transpose(0, 1))

    width = xmax-xmin
    height = ymax-ymin
    rect = patches.Rectangle((xmin,ymin),width,height,linewidth=3,edgecolor=color,facecolor='none')
    ax.add_patch(rect)
    infos.insert( 0, f"Step: {count}" )
    ax.set_title( ', '.join(infos) )
    save_path = os.path.join(save_path, f'{prefix}{count}.png')
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

def make_movie(in_dir, in_prefix, out_dir, total_frames, del_source_images=False):
    tested = 0
    while os.path.isfile(f'{out_dir}/movie_{tested}.gif'): tested += 1
    images = []
    for count in range(1, total_frames+1):
        images.append(imageio.imread(f'{in_dir}/{in_prefix}{count}.png'))

    imageio.mimsave(f'{out_dir}/movie_{tested}.gif', images)

    if del_source_images:
        for count in range(1, total_frames):
            os.remove(f'{in_dir}/{in_prefix}{count}.png')

def extract(index, loader):
    extracted = loader[index]
    ground_truth_boxes =[]
    for ex in extracted:
        img = ex[0]
        bndbox = ex[1][0]
        size = ex[1][1]
        xmin = ( float(bndbox['xmin']) /  float(size['width']) ) * 224
        xmax = ( float(bndbox['xmax']) /  float(size['width']) ) * 224

        ymin = ( float(bndbox['ymin']) /  float(size['height']) ) * 224
        ymax = ( float(bndbox['ymax']) /  float(size['height']) ) * 224

        ground_truth_boxes.append([xmin, xmax, ymin, ymax])
    return img, ground_truth_boxes


def voc_ap(rec, prec, voc2007=True):
    if voc2007:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([1.0], prec, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def prec_rec_compute(bounding_boxes, gt_boxes, ovthresh):
    nd = len(bounding_boxes)
    npos = nd
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    d = 0

    for index in range(len(bounding_boxes)):
        box1 = bounding_boxes[index]
        box2 = gt_boxes[index][0]
        x11, x21, y11, y21 = box1[0], box1[1], box1[2], box1[3]
        x12, x22, y12, y22 = box2[0], box2[1], box2[2], box2[3]

        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area

        if iou > ovthresh:
            tp[d] = 1.0
        else:
            fp[d] = 1.0
        d += 1

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    return prec, rec


def compute_ap_and_recall(all_bdbox, all_gt, ovthresh):
    prec, rec = prec_rec_compute(all_bdbox, all_gt, ovthresh)
    ap = voc_ap(rec, prec, True)
    return ap, rec[-1]


def eval_stats_at_threshold( all_bdbox, all_gt, thresholds=[0.4, 0.5, 0.6]):
    stats = {}
    for ovthresh in thresholds:
        ap, recall = compute_ap_and_recall(all_bdbox, all_gt, ovthresh)
        stats[ovthresh] = {'ap': ap, 'recall': recall}
    stats_df = pd.DataFrame.from_records(stats)*100
    return stats_df


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_tensor(tensor_func, size):
    tensor = tensor_func(size)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor

def extract_features(feature_extractor, image, dtype=FloatTensor):
    global transform
    #image = transform(image)
    image = image.view(1,*image.shape)
    image = Variable(image).type(dtype)
    if use_cuda:
        image = image.cuda()
    feature = feature_extractor(image)
    #print("Feature shape : "+str(feature.shape))
    return feature.data

def _get_feature_extractor(use_cuda, network):
    feature_extractor = FeatureExtractor(network=network)
    feature_extractor.eval()
    if use_cuda: feature_extractor = feature_extractor.cuda()
    return feature_extractor

def get_feature_extractor(use_cuda, network='vgg16'):
    fe = _get_feature_extractor(use_cuda, network)
    feature_extractor = lambda img: extract_features(fe,img)
    feature_extractor_dim = 25088 if network=='vgg16' else 512*4 #resnet50
    return feature_extractor, feature_extractor_dim

def get_vectorized_env(
        env_class,
        image_loader,
        feature_extractor,
        fe_out_dim,
        n_envs,
        need_render=False
    ):
    args = {
        'image_loader': image_loader,
        'feature_extractor': feature_extractor,
        'feature_extractor_dim': fe_out_dim,
        'need_render' : need_render
    }
    env = make_vec_env(
                env_id=env_class,
                n_envs=n_envs,
                env_kwargs=args
            )
    # env = VecCheckNan(env, raise_exception=True)
    return env

def unwrap_vec_env( vec_env ):
    return vec_env.unwrapped.envs[0].unwrapped
