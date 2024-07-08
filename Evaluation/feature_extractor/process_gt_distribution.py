import os
import torch
from tqdm import tqdm
import numpy as np
from utils.bvh_motion import Motion
from train_classifier import Classifier
from process_train_data import process_file
import multiprocessing
from multiprocessing import Pool

def make_data(bvh_paths):
    with Pool(multiprocessing.cpu_count()) as p:
        data_list = list(tqdm(p.imap(process_file, bvh_paths), total=len(bvh_paths)))
    data_list = [data for data in data_list if data is not None]
    return data_list


if __name__ == '__main__':
    
    clip_len = 60
    
    data_folders = [
        ["./data/raw", "style_skeleton"],
        ["./data/raw_mann", "ybot_skeleton"]
    ]
    
    bvh_paths = []
    for data_folder, skeleton_config in data_folders:
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if '.bvh' in file:
                    bvh_paths.append([os.path.join(root, file), skeleton_config])
    
    device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
    classifier = Classifier(in_feat=45, class_num=100, device=device)
    save_dict = torch.load('100style_position_classifier.pth', map_location=device)
    classifier.load_state_dict(save_dict['state_dict'])
    classifier.to(device)
    classifier.eval()
    styleList = save_dict['class_names']
    
    style_files = {}
    for style in styleList:
        style_files[style] = []
    for bvh_path, skeleton_config in bvh_paths:
        style_name = bvh_path.split('/')[-1].split('_')[0]
        style_files[style_name].append([bvh_path, skeleton_config])
    
    style_features_dic = {}
    for style in styleList:
        style_bvh_files = style_files[style]
        motion_data_list = make_data(style_bvh_files)
        style_features = []
        for data in motion_data_list:
            local_position = data['local_position']
            frame_num = local_position.shape[0]
            local_position = local_position.reshape(frame_num, -1)
            clip_indices = np.arange(0, frame_num - clip_len + 1, clip_len)[:, None] + np.arange(clip_len)
            model_input = local_position[clip_indices]
            model_input = torch.tensor(model_input, dtype=torch.float32).to(device)
            with torch.no_grad():
                feats, out = classifier(model_input.permute(0, 2, 1))
            style_features.append(feats.cpu().numpy())
        style_features_all = np.concatenate(style_features, axis=0)
        mu, sigma = np.mean(style_features_all, axis=0), np.cov(style_features_all, rowvar=False)
        style_features_dic[style] = [mu, sigma]
    np.savez('100style_position_distribution.npz', style_features_dic=style_features_dic)