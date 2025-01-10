import sys
sys.path.append('./')

import os
import pickle
import numpy as np
import utils.motion_modules as motion_modules
import style_helper as style100
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from utils.bvh_motion import Motion


def load_contact(contact_path):
    contacts = []
    for line in open(contact_path, 'r'):
        contacts.append([int(val) for val in line.strip().split()])
    return np.array(contacts)


def mirror_text(statement):
    statement = statement.lower()
    temp_replacement = "_*temp*_"
    statement = statement.replace("left", temp_replacement)
    statement = statement.replace("right", "left")    
    statement = statement.replace(temp_replacement, "right")
    return statement


def extract_traj(root_positions, forwards, smooth_kernel=[5, 10]):
    traj_trans, traj_angles, traj_poses = [], [], []
    FORWARD_AXIS = np.array([[0, 0, 1]]) # OpenGL system
    
    for kernel_size in smooth_kernel:
        smooth_traj = gaussian_filter1d(root_positions[:, [0, 2]], kernel_size, axis=0, mode='nearest')
        traj_trans.append(smooth_traj)
        
        forward = gaussian_filter1d(forwards, kernel_size, axis=0, mode='nearest')
        forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
        angle = np.arctan2(forward[:, 2], forward[:, 0])
        traj_angles.append(angle)
        
        v0s = FORWARD_AXIS.repeat(len(forward), axis=0)
        a = np.cross(v0s, forward)
        w = np.sqrt((v0s**2).sum(axis=-1) * (forward**2).sum(axis=-1)) + (v0s * forward).sum(axis=-1)
        between_wxyz = np.concatenate([w[...,np.newaxis], a], axis=-1)
        between = R.from_quat(between_wxyz[..., [1, 2, 3, 0]]).as_quat()[..., [3, 0, 1, 2]]
        
        traj_poses.append(between)
    
    return traj_trans, traj_angles, traj_poses
    
    
def process_motion(motion, use_scale=True):
    if use_scale:
        motion = motion_modules.scaling(motion, scaling_factor=0.01) # cm -> m
    motion = motion_modules.root(motion)
    motion = motion_modules.on_ground(motion)
    # _, forwards = motion_modules.extract_forward(motion, np.arange(motion.frame_num),
    #                                                         style100.left_shoulder_name, style100.right_shoulder_name, 
    #                                                         style100.left_hip_name, style100.right_hip_name, return_forward=True)
    _, forwards = motion_modules.extract_forward_hips(motion, np.arange(motion.frame_num), style100.left_hip_name, style100.right_hip_name, return_forward=True)
    traj_trans, traj_angles, traj_poses = extract_traj(motion.global_positions[:, 0], forwards, smooth_kernel=[5, 10])
    return motion, {
        'filepath': motion.filepath,
        'local_joint_rotations': motion.rotations,
        'global_root_positions': motion.global_positions[:, 0],
        'traj': traj_trans,
        'traj_angles': traj_angles,
        'traj_pose': traj_poses
    }

if __name__ == '__main__':
    
    data_root = 'data/100STYLE_mixamo'
    process_batch = os.path.join(data_root, 'simple')
    export_path = 'data/pkls/100style.pkl'
    style_metas = style100.get_info(data_root, meta_file='Dataset_List.csv')
    
    data_list = {
        'parents': None,
        'offsets': None,
        'names': None,
        'motions': []
    }
    
    bvh_files = []
    for root, dirs, files in os.walk(process_batch):
        for file in files:
            if file.endswith('.bvh'):
                bvh_files.append(os.path.join(root, file))
    bvh_files = sorted(bvh_files)
    
    for bvh_path in tqdm(bvh_files):
        
        style_name = os.path.basename(bvh_path).replace('.bvh', '').split('')[0]
        action_label = os.path.basename(bvh_path).replace('.bvh', '').split('')[-1]
        
        if style_name not in style_metas.keys():
            continue
        meta_info = style_metas[style_name]
        start_idx, end_idx = meta_info['framecuts'][action_label]
        
        start_idx, end_idx = start_idx // 2, end_idx // 2
                
        motion = Motion.load_bvh(bvh_path)[start_idx:end_idx]
        motion, motion_data = process_motion(motion)
        motion_data['text'] = meta_info['description'].lower()
        motion_data['style'] = style_name
        data_list['motions'].append(motion_data)
        
        if meta_info['is_symmetric']:
            mirror_motion = motion.copy()
            names = mirror_motion.names
            l_names = sorted([val for val in names if 'left' in val.lower()])
            r_names = sorted([val for val in names if 'right' in val.lower()])
            l_joint_idxs, r_joint_idxs = [names.index(name) for name in l_names], [names.index(name) for name in r_names]
            mirror_motion = motion_modules.mirror(mirror_motion, l_joint_idxs, r_joint_idxs)
            _, mirror_motion_data = process_motion(mirror_motion, use_scale=False)
            mirror_motion_data['filepath'] = mirror_motion_data['filepath'].replace('.bvh', '.mirror.bvh')
            mirror_motion_data['text'] = mirror_text(meta_info['description'])
            mirror_motion_data['style'] = style_name
            data_list['motions'].append(mirror_motion_data)
        print('Finish processing %s' % bvh_path)
    
    T_rotation = np.zeros((1, motion.rotations.shape[1], motion.rotations.shape[2]))
    T_rotation[..., 0] = 1
    T_position = np.zeros((1, motion.positions.shape[1], motion.positions.shape[2]))
    motion.rotations = T_rotation
    motion.positions = T_position
    data_list['T_pose'] = motion
    
    pickle.dump(data_list, open(export_path, 'wb'))
    print('Finish exporting %s' % export_path)
