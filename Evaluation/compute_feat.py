import os
import torch
from tqdm import tqdm
import numpy as np
from feature_extractor.train_classifier import Classifier, window_size
from feature_extractor.process_train_data import styleList, process_file
import multiprocessing
from multiprocessing import Pool

def make_data(bvh_paths):
    with Pool(multiprocessing.cpu_count()) as p:
        data_list = list(tqdm(p.imap(process_file, bvh_paths), total=len(bvh_paths)))
    data_list = [data for data in data_list if data is not None]
    return data_list


if __name__ == '__main__':
    
    clip_len = window_size
    
    process_folder = [
        ["data/exp1_mann+dp", "ybot_skeleton"],
        ["data/exp1_mann+lp", "ybot_skeleton"],
        ["data/exp1_moglow", "style_skeleton"],
        ["data/exp1_ours", "style_skeleton"],
        ["data/exp1_ours_mlp", "style_skeleton"],
        ["data/exp2_mann+dp", "ybot_skeleton"],
        ["data/exp2_mann+lp", "ybot_skeleton"],
        ["data/exp2_moglow", "style_skeleton"],
        ["data/exp2_ours", "style_skeleton"],
    ]
    
    bvh_paths = []
    for data_folder, skeleton_config in process_folder:
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if '.bvh' in file:
                    bvh_paths.append([os.path.join(root, file), skeleton_config])
    
    motion_data_list = make_data(bvh_paths)
    local_positions = [data['local_position'] for data in motion_data_list]
    
    device = torch.device('cpu')
    classifier = Classifier(in_feat=45, class_num=100, device=device)
    save_dict = torch.load('feature_extractor/100style_position_classifier.pth', map_location=device)
    classifier.load_state_dict(save_dict['state_dict'])
    classifier.eval()
    style_names = save_dict['class_names']
    
    for data in motion_data_list:
        save_path = data['file_path'].replace('.bvh', '_predicted.npz')
        if os.path.exists(save_path):
            continue
        local_position = data['local_position']
        frame_num = local_position.shape[0]
        local_position = local_position.reshape(frame_num, -1)
        clip_indices = np.arange(0, frame_num - clip_len + 1, clip_len)[:, None] + np.arange(clip_len)
        model_input = local_position[clip_indices]
        model_input = torch.tensor(model_input, dtype=torch.float32).to(device)
        with torch.no_grad():
            feats, out = classifier(model_input.permute(0, 2, 1))
            _, example_predicted = torch.max(out.data, 1)
            pre_style_name = [style_names[idx] for idx in example_predicted.cpu().numpy()]
        
        feats_numpy = feats.cpu().numpy()
        np.savez(save_path, feats=feats_numpy, predicted=pre_style_name)
        print('Processed:', save_path)
