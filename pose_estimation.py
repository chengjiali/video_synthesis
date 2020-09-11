from tqdm import tqdm
from pathlib import Path

import cv2
import torch
import numpy as np


def get_pose_estimation_model():
    ''' Get the pose estimation model '''

    weight_name = openpose_dir.joinpath('network/weight/pose_model.pth')
    model = get_model('vgg19')
    model.load_state_dict(torch.load(weight_name))
    model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()

    return model

def pose_estimation(model, video_dir, start_idx=0, length=1000):
    ''' Perform pose estimation
    Parameter:
    ----------
        video_dir: directory to video
        start_idx: start index of the images to be estimated
        length: number of frames to use
    '''

    video_dir = video_dir.joinpath(video_dir)
    frame_dir = video_dir.joinpath('frames')
    pose_dir = video_dir.joinpath('pose')
    pose_dir.mkdir(exist_ok=True)

    for idx in tqdm(range(start_idx, start_idx + length)):
        frame_path = frame_dir.joinpath(f'img_{idx:04d}.png')
        img = cv2.imread(frame_path.as_posix())
        shape_dst = np.min(img.shape[:2])
        oh = (img.shape[0] - shape_dst) // 2
        ow = (img.shape[1] - shape_dst) // 2

        img = img[oh:oh+shape_dst, ow:ow+shape_dst]
        img = cv2.resize(img, (512, 512))
        multiplier = get_multiplier(img)
        with torch.no_grad():
            paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')
        r_heatmap = np.array([remove_noise(ht) for ht in 
                             heatmap.transpose(2, 0, 1)[:-1]]).transpose(1, 2, 0)
        heatmap[:, :, :-1] = r_heatmap
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        label = get_pose(param, heatmap, paf)
        cv2.imwrite(pose_dir.joinpath(f'label_{idx:04d}.png').as_posix(), label)
        
    torch.cuda.empty_cache()

def get_outputs(img, model, preprocess):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """
    inp_size = cfg.DATASET.IMAGE_SIZE

    # padding
    im_croped, im_scale, real_shape = im_transform.crop_with_factor(
        img, inp_size, factor=cfg.MODEL.DOWNSAMPLE, is_ceil=True)

    if preprocess == 'rtpose':
        im_data = rtpose_preprocess(im_croped)

    elif preprocess == 'vgg':
        im_data = vgg_preprocess(im_croped)

    elif preprocess == 'inception':
        im_data = inception_preprocess(im_croped)

    elif preprocess == 'ssd':
        im_data = ssd_preprocess(im_croped)

    batch_images= np.expand_dims(im_data, 0)

    # several scales as a batch
    batch_var = torch.from_numpy(batch_images).cuda().float()
    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
    heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

    return paf, heatmap, im_scale


if __name__ == "__main__":

    model = get_pose_estimation_model()
    pose_estimation(model, '../data/source', 0, 1000)
    pose_estimation(model, '../data/target', 0, 1000) 
