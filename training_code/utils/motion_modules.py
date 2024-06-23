import sys
sys.path.append('./')

import numpy as np

from utils.bvh_motion import Motion
from scipy.spatial.transform import Rotation as R

'''
Rotation with Quaternion (w, x, y, z)
'''

def remove_joints(motion: Motion, remove_key_words):
    remove_name, remove_idxs = [], []
    for name_idx, name in enumerate(motion.names):
        for key_word in remove_key_words:
            if key_word in name.lower():
                remove_name.append(name)
                remove_idxs.append(name_idx)
                
    # remove child joints
    for idx in range(len(motion.names)):
        if motion.parents[idx] in remove_idxs and idx not in remove_idxs:
            remove_name.append(motion.names[idx])
            remove_idxs.append(idx)
    
    ori_name, ori_parents = motion.names.copy(), motion.parents.copy()
    removed_rotation = motion.rotations[:, remove_idxs]
    removed_offset = motion.offsets[remove_idxs]
    motion.rotations = np.delete(motion.rotations, remove_idxs, axis=1)
    motion.positions = np.delete(motion.positions, remove_idxs, axis=1)
    motion.offsets = np.delete(motion.offsets, remove_idxs, axis=0)
    motion.names = np.delete(motion.names, remove_idxs, axis=0)
    
    # update parents
    motion.parents = np.zeros(len(motion.names), dtype=np.int32)
    for idx in range(len(motion.names)):
        parent_name = motion.parent_names[motion.names[idx]]
        if parent_name is None:
            motion.parents[idx] = -1
        else:
            motion.parents[idx] = np.where(motion.names == parent_name)[0][0]
    motion.removed_joints.append([removed_rotation, removed_offset, remove_idxs, ori_parents])
    motion.opt_history.append('remove_joints')
    return motion


def scaling(motion: Motion, scaling_factor=None):
    if scaling_factor is None:
        t_pose = motion.get_T_pose()
        heights = t_pose[:, 1]
        height_diff = np.max(heights) - np.min(heights)
        scaling_factor = height_diff
    motion.positions *= scaling_factor
    motion.offsets *= scaling_factor
    motion.end_offsets *= scaling_factor
    motion.scaling_factor = scaling_factor
    motion.opt_history.append('scaling')
    return motion


def mirror(motion: Motion, l_joint_idxs, r_joint_idxs):
    ori_rotations = motion.rotations.copy()
    # mirror root trajectory
    motion.positions[:, :, 0] *= -1
    
    # mirror joint rotations
    motion.rotations[:, l_joint_idxs] = ori_rotations[:, r_joint_idxs]
    motion.rotations[:, r_joint_idxs] = ori_rotations[:, l_joint_idxs]
    motion.rotations[:, :, [2, 3]] *= -1
    
    # mirror hip rotations
    motion.rotations[:, 0] *= -1 
    motion.opt_history.append('mirror')
    return motion


def on_ground(motion: Motion):
    global_pos = motion.global_positions
    lowest_height = np.min(global_pos[:, :, 1])
    motion.positions[:, :, 1] -= lowest_height
    motion.opt_history.append('on_ground')
    return motion


def root(motion: Motion, given_pos=None, return_pos=False):
    root_init_pos = motion.positions[0, 0, [0, 2]] if given_pos is None else given_pos
    motion.positions[:, 0, [0, 2]] -= root_init_pos
    motion.opt_history.append('root')
    if return_pos:
        return motion, root_init_pos
    else:
        return motion


def temporal_scale(motion: Motion, scale_factor: int):
    motion.positions = motion.positions[::scale_factor]
    motion.rotations = motion.rotations[::scale_factor]
    motion.frametime *= scale_factor
    motion.opt_history.append('temporal_scale')
    return motion


'''
frame_idx: int or list
'''
def extract_forward(motion: Motion, frame_idx, left_shoulder_name, right_shoulder_name, left_hip_name, right_hip_name, return_forward=False):
    if type(frame_idx) is int:
        frame_idx = [frame_idx]
    
    names = list(motion.names)
    try:
        l_s_idx, r_s_idx = names.index(left_shoulder_name), names.index(right_shoulder_name)
        l_h_idx, r_h_idx = names.index(left_hip_name), names.index(right_hip_name)
    except:
        raise Exception('Cannot find joint names, please check the names of Hips and Shoulders in the bvh file.')
    global_pos = motion.update_global_positions()
    
    upper_across = global_pos[frame_idx, l_s_idx, :] - global_pos[frame_idx, r_s_idx, :]
    lower_across = global_pos[frame_idx, l_h_idx, :] - global_pos[frame_idx, r_h_idx, :]
    across = upper_across / np.sqrt((upper_across**2).sum(axis=-1))[...,np.newaxis] + lower_across / np.sqrt((lower_across**2).sum(axis=-1))[...,np.newaxis]
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward_angle = np.arctan2(forward[:, 2], forward[:, 0])
    if return_forward:
        return forward_angle, forward
    return forward_angle


def extract_path_forward(motion: Motion, start_frame=0, end_frame=60):
    if motion.frame_num < end_frame:
        end_frame = motion.frame_num - 1
    root_xz = motion.positions[:, 0, [0, 2]]
    root_xz_offset = root_xz[end_frame] - root_xz[start_frame]
    forward_angle = np.arctan2(root_xz_offset[1], root_xz_offset[0])
    return forward_angle 


def rotate(motion: Motion, given_angle=None, axis='y', return_angle=False):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if len(given_angle) > 1:
        assert len(given_angle) == motion.frame_num
        rot_vec = np.zeros((motion.frame_num, 3))
        rot_vec[:, axis_map[axis]] = given_angle
        given_rotation = R.from_rotvec(rot_vec)
    else:
        rot_vec = np.zeros(3)
        rot_vec[axis_map[axis]] = given_angle
        given_rotation = R.from_rotvec(rot_vec)
    ori_root_R_xyzw = motion.rotations[:, 0][..., [1, 2, 3, 0]]
    ori_root_pos = motion.positions[:, 0]
    rotated_root = (given_rotation * R.from_quat(ori_root_R_xyzw)).as_quat()[..., [3, 0, 1, 2]]
    rotated_root_pos = given_rotation.apply(ori_root_pos)
    motion.rotations[:, 0] = rotated_root
    motion.positions[:, 0] = rotated_root_pos
    motion.opt_history.append('rotate_%s_%s' % (axis, given_angle))
    return motion


if __name__ == '__main__':
    bvh_path = 'data/raw/multi-subject/test/JL-JR-res-subject10_faceZ.bvh'
    motion = Motion.load_bvh(bvh_path)
    
    import os
    # copy from original path to new path
    os.system('cp %s vis/module_test/copy.bvh' % bvh_path)
    motion.plot(save_path='vis/module_test/skeleton.png')
    motion.export('vis/module_test/export.bvh', order='XZY')
    
    # On the Ground
    motion_on_ground = on_ground(motion.copy())
    # motion_on_ground.export('vis/module_test/on_ground.bvh')
    
    # FK
    motion.update_global_positions()
    position_np = motion.global_positions
    from utils.nn_transforms import neural_FK
    import torch
    rotation_tensor = torch.from_numpy(motion.rotations).float()
    offset_tensor = torch.from_numpy(motion.offsets).float()
    root_pos_tensor = torch.from_numpy(motion.positions[:, 0]).float()
    parents_tensor = torch.tensor([motion.parents])
    position_tensor = neural_FK(rotation_tensor, offset_tensor, root_pos_tensor, parents_tensor).numpy()
    error = np.mean(np.abs(position_np - position_tensor))
    print('FK error: %.4f' % error)
    
    # mirror
    names = motion.names
    l_names = sorted([val for val in names if val.lower()[0] is 'l'])
    r_names = sorted([val for val in names if (val.lower()[0] is 'r' and val.lower() != 'root')])
    l_joint_idxs, r_joint_idxs = [names.index(name) for name in l_names], [names.index(name) for name in r_names]
    motion_mirror = mirror(motion.copy(), l_joint_idxs, r_joint_idxs)
    # motion_mirror.export('vis/module_test/mirror.bvh')
    
    # remove unsupported joints
    remove_key_words = ['index', 'middle', 'ring', 'pinky', 'thumb']
    motion_remove = remove_joints(motion.copy(), remove_key_words)
    # motion_remove.export('vis/module_test/remove.bvh')
    
    # scaling
    motion_scaled = scaling(motion.copy())
    # motion_scaled.export('vis/module_test/scaled.bvh')
    
    # rotate
    forward_angle = extract_forward(motion.copy(), 0, 'lshoulder', 'rshoulder', 'lfemur', 'rfemur')
    motion_rotate = rotate(motion.copy(), given_angle=forward_angle, axis='y')
    motion_rotate.export('vis/module_test/redirection.bvh')
    
    # rotate refer to the path
    path_forward_angle = extract_path_forward(motion.copy(), 0, 60)
    motion_rotate_path = rotate(motion.copy(), given_angle=path_forward_angle, axis='y')
    motion_rotate_path.export('vis/module_test/redirection_path.bvh')
    
    # root
    motion_root = root(motion.copy())
    # motion_root.export('vis/module_test/root.bvh')
    