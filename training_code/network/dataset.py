import sys
sys.path.append('./')

import torch
import pickle
import random
import numpy as np

import utils.nn_transforms as nn_transforms
from scipy.ndimage import gaussian_filter1d

from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    '''
    rot_req: str, rotation format, 'q'|'6d'|'euler'
    window_size: int, window size for each clip
    '''
    def __init__(self, pkl_path, rot_req, offset_frame, past_frame, future_frame, dtype=np.float32, limited_num=-1):
        self.pkl_path, self.rot_req,self.dtype = pkl_path, rot_req, dtype
        window_size = past_frame + future_frame
        self.past_frame = past_frame
        self.rotations_list, self.root_pos_list = [], []
        self.local_conds = {'traj_pose': [], 'traj_trans': []}
        self.global_conds = {'style': []}
        
        self.rot_feat_dim = {'q': 4, '6d': 6, 'euler': 3}
        self.reference_frame_idx = past_frame
        
        data_source = pickle.load(open(pkl_path, 'rb'))

        self.T_pose = data_source['T_pose']
        frame_nums = []
        item_frame_indices_list = [] # store the frame index for each clip. shape = 1 + window_size, 1 represent the motion idx
        motion_idx = 0
        for motion_item in tqdm(data_source['motions'][:limited_num]):
            frame_num = motion_item['local_joint_rotations'].shape[0]
            if frame_num < window_size:
                continue
            frame_nums.append([motion_item['style'], frame_num])
            self.rotations_list.append(motion_item['local_joint_rotations'].astype(dtype))
            self.root_pos_list.append(motion_item['global_root_positions'].astype(dtype))
            
            self.local_conds['traj_pose'].append(np.array([item for item in motion_item['traj_pose']], dtype=dtype))
            self.local_conds['traj_trans'].append(np.array([item for item in motion_item['traj']], dtype=dtype))
            
            self.global_conds['style'].append(motion_item['style'])

            clip_indices = np.arange(0, frame_num - window_size + 1, offset_frame)[:, None] + np.arange(window_size)
            clip_indices_with_idx = np.hstack((np.full((len(clip_indices), 1), motion_idx, dtype=clip_indices.dtype), clip_indices))            
            item_frame_indices_list.append(clip_indices_with_idx)
            motion_idx += 1
            
        self.item_frame_indices = np.concatenate(item_frame_indices_list, axis=0)
    
        self.joint_num, self.per_rot_feat = self.rotations_list[0].shape[-2], self.rot_feat_dim[rot_req]
        self.traj_aug_indexs1 = list(range(self.local_conds['traj_pose'][0].shape[0]))
        self.traj_aug_indexs2 = list(range(self.local_conds['traj_trans'][0].shape[0]))
        self.mask = np.ones(window_size - self.reference_frame_idx, dtype=bool)
        self.style_set = sorted(list(set(self.global_conds['style'])))
        print('Dataset loaded, trained with %d clips, %d frames, %d mins in total' % (len(frame_nums), sum([item[1] for item in frame_nums]), sum([item[1] for item in frame_nums])/30/60))
        
        
    def __len__(self):
        return len(self.item_frame_indices)
    
    def __getitem__(self, idx):
        item_frame_indice = self.item_frame_indices[idx]
        motion_idx, frame_indices = item_frame_indice[0], item_frame_indice[1:]
        
        rotations = self.rotations_list[motion_idx][frame_indices].copy()
        root_pos = self.root_pos_list[motion_idx][frame_indices].copy()
        root_pos[:, [0, 2]] -= root_pos[self.reference_frame_idx-1:self.reference_frame_idx, [0, 2]]
        traj_rotation = self.local_conds['traj_pose'][motion_idx][random.choice(self.traj_aug_indexs1), frame_indices].copy()

        traj_pos = root_pos[:, [0, 2]].copy()
        random_option = np.random.random()
        # Random filtering trajectory. Relieve the out-of-distribution(OOD) problem during runtime
        if random_option <0.5:
            pass
        elif random_option < 0.75:
            traj_pos = gaussian_filter1d(traj_pos, 5, axis=0)
        else:
            traj_pos = gaussian_filter1d(traj_pos, 10, axis=0)
            
        traj_pos -= traj_pos[self.reference_frame_idx-1:self.reference_frame_idx]

        traj_rotation = traj_rotation[self.reference_frame_idx:]
        traj_pos = traj_pos[self.reference_frame_idx:]
        
        # Random rotation augmentation 
        rotation_xyzw, traj_rotation_xyzw = rotations[..., [1, 2, 3, 0]], traj_rotation[..., [1, 2, 3, 0]]
        
        theta = np.repeat(np.random.uniform(0, 2*np.pi), rotations.shape[0]).astype(self.dtype)
        rot_vec = R.from_rotvec(np.pad(theta[..., np.newaxis], ((0, 0), (1, 1)), 'constant', constant_values=0))
        rotations[:, 0] = (rot_vec*R.from_quat(rotation_xyzw[:, 0])).as_quat()[..., [3, 0, 1, 2]]
        traj_rotation = (rot_vec[self.reference_frame_idx:]*R.from_quat(traj_rotation_xyzw)).as_quat()[..., [3, 0, 1, 2]]
        root_pos = rot_vec.apply(root_pos).astype(self.dtype)
        traj_pos_3d = np.zeros((traj_pos.shape[0], 3), dtype=self.dtype)
        traj_pos_3d[:, [0, 2]] = traj_pos
        traj_pos = rot_vec[self.reference_frame_idx:].apply(traj_pos_3d)[:, [0, 2]].astype(self.dtype)

        rotations = nn_transforms.get_rotation(rotations.astype(self.dtype), self.rot_req)
        traj_rotation = nn_transforms.get_rotation(traj_rotation.astype(self.dtype), self.rot_req)
        
        root_pos_extra_dim = np.zeros((root_pos.shape[0], 1, self.per_rot_feat - 3), dtype=self.dtype)
        root_pos_extra_dim = torch.from_numpy(np.concatenate((root_pos[:, np.newaxis], root_pos_extra_dim), axis=2, dtype=self.dtype))
        rotations_with_root = torch.cat((rotations, root_pos_extra_dim), axis=1)
        
        future_motion = rotations_with_root[self.reference_frame_idx:]
        past_motion = rotations_with_root[:self.reference_frame_idx]
    
        style_idx = float(self.style_set.index(self.global_conds['style'][motion_idx]))
        
        return {
            'data': future_motion,
            'conditions': {
                'past_motion': past_motion,
                'traj_pose': traj_rotation,
                'traj_trans': traj_pos,
                'style': self.global_conds['style'][motion_idx],
                'style_idx': style_idx,
                'mask': self.mask
            }
        }     



# Case test for the dataset class
if __name__ == '__main__':
    
    import time
    import torch
    
    pkl_path = 'data/pkls/100style.pkl'
    rot_req = '6d'
    train_data = MotionDataset(pkl_path, rot_req, offset_frame=1, past_frame=10, future_frame=45, dtype=np.float32)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1024, shuffle=True, num_workers=8, drop_last=False, pin_memory=True)
    
    do_export_test = False
    do_loop_test = True
    
    if do_export_test:
        T_pose = train_data.T_pose
        data = train_dataloader.__iter__().__next__()
        rotations = nn_transforms.repr6d2quat(data['rotations']).numpy()
        root_pos = data['root_pos'].numpy()
         
        for i in range(10):
            T_pose_template = T_pose.copy()
            T_pose_template.rotations = rotations[i]
            T_pose_template.positions = np.zeros((rotations[i].shape[0], T_pose_template.positions.shape[1], T_pose_template.positions.shape[2]))
            T_pose_template.positions[:, 0] = root_pos[i]
            T_pose_template.export('save/visualization/example_bvh/%s.bvh' % i, save_ori_scal=True)  
             
    if do_loop_test:
        times = []
        start_time = time.time()
        for data in train_dataloader:
            end_time = time.time()
            print('Data loading time for each iteration: %ss' % (end_time - start_time))
            times.append(end_time - start_time)
            start_time = end_time
        avg_time = sum(times) / len(times)
        print('Average data loading time: %ss' % avg_time)
        print('Entire data loading time: %ss' % sum(times))
    