"""
This script is written to handle the motion data in BVH format, including loading, saving, and processing.
It can work with motion_module.py to operate the motion class in different ways.
Please contact the mailto:mingyis@connect.hku.hk if you meet any issues.
Author: Mingyi Shi
Date: 06/01/2024
"""

import re
import os
import copy
import numpy as np

from scipy.spatial.transform import Rotation as R

'''
Rotation with Quaternion (w, x, y, z)
'''

channelmap = {'Xrotation' : 'x', 'Yrotation' : 'y', 'Zrotation' : 'z'}
channelmap_inv = {'x': 'Xrotation', 'y': 'Yrotation', 'z': 'Zrotation', 'X': 'Xrotation', 'Y': 'Yrotation', 'Z': 'Zrotation'}
ordermap = {'x' : 0, 'y' : 1, 'z' : 2, 'X' : 0, 'Y' : 1, 'Z' : 2}

    
def clean_quat(quat):
    frame_num, joint_num = quat.shape[:2]
    for f_idx in range(1, frame_num):
        for j_idx in range(joint_num):
            last_quat = quat[f_idx - 1, j_idx]
            cur_quat = quat[f_idx, j_idx]
            dis_1 = ((last_quat - cur_quat) ** 2).sum()
            dis_2 = ((last_quat + cur_quat) ** 2).sum()
            if dis_2 < dis_1:
                quat[f_idx, j_idx] = -cur_quat
    return quat


class Motion:
    """
    The Motion class handles the representation and transformation of motion capture data.
    """
    def __init__(self, rotations: np.ndarray, positions: np.ndarray, 
                 offsets: np.ndarray, parents: np.ndarray, 
                 names, frametime: float, end_offsets=None, _global_positions=None, filepath=None, 
                 sampling_factor=1.0, scaling_factor=1.0, removed_joints=None, opt_history=None):
        
        if not (rotations.shape[0] == positions.shape[0]):
            raise ValueError("The shape of rotations, positions must be compatible.")
        if not (rotations.shape[1] == offsets.shape[0] == parents.shape[0] == len(names)):
            raise ValueError("The shape of rotations, offsets, parents, names must be compatible.")
        
        self._rotations = rotations
        self._positions = positions
        self._offsets = offsets
        self._global_positions = _global_positions  
        
        self.end_offsets = end_offsets
        self.parents = parents
        self.names = names
        self.frametime = frametime
        self.frame_num, self.joint_num = rotations.shape[:2]
        self.filepath = filepath
        self.sampling_factor = sampling_factor
        
        # opetation parameters - save for exporting in same format
        self.scaling_factor = scaling_factor
        self.removed_joints = [] if removed_joints is None else removed_joints
        self.opt_history = [] if opt_history is None else opt_history
        
        self.parent_names = {}
        for i in range(len(self.names)):
            if self.parents[i] == -1:
                self.parent_names[self.names[i]] = None
            else:
                self.parent_names[self.names[i]] = self.names[self.parents[i]]
              
    @property
    def rotations(self):
        return self._rotations

    @rotations.setter
    def rotations(self, value):
        self._rotations = value
        self.frame_num, self.joint_num = value.shape[:2]
        self._global_position = None
        
    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value
        self._global_position = None
        
    @property
    def offsets(self):
        return self._offsets

    @offsets.setter
    def offsets(self, value):
        self._offsets = value
        self._global_position = None

    @property
    def global_positions(self):
        if self._global_positions is None:
            self._global_positions = self.get_joint_positions()
        return self._global_positions

    @property
    def shape(self): return (self.frame_num, self.joint_num)

    def update_global_positions(self):
        self._global_positions = self.get_joint_positions()
        return self._global_positions


    # if the frame num is too small, extent it
    # repeat: repeat the last frame
    # mirror: mirror the motion
    def extend_frames(self, target_frame_num, update_method='repeat'):
        if self.frame_num >= target_frame_num:
            return self
        else:
            additional_frames_needed = target_frame_num - self.frame_num
            if update_method == 'repeat':
                repeat_times = additional_frames_needed // self.frame_num + 1
                self.rotations = np.tile(self.rotations, (repeat_times, 1))[:target_frame_num]
                self.positions = np.tile(self.positions, (repeat_times, 1))[:target_frame_num]
            elif update_method == 'mirror':
                mirrored_rotations = np.concatenate([self.rotations, self.rotations[-2::-1]], axis=0)
                mirrored_positions = np.concatenate([self.positions, self.positions[-2::-1]], axis=0)
                repeat_times = additional_frames_needed // len(mirrored_rotations) + 1
                self.rotations = np.tile(mirrored_rotations, (repeat_times, 1))[:target_frame_num]
                self.positions = np.tile(mirrored_positions, (repeat_times, 1))[:target_frame_num]
            else:
                raise ValueError("The update method must be 'repeat' or 'mirror'.")
                self.frame_num = target_frame_num

    def get_T_pose(self):
        t_pose = np.zeros((self.joint_num, 3))
        for joint_idx, parent_idx in enumerate(self.parents):
            t_pose[joint_idx] = self.offsets[joint_idx] + t_pose[parent_idx]
        return t_pose

    def get_bone_lengths(self):
        bone_lengths = np.zeros(self.joint_num)
        for joint_idx, parent_idx in enumerate(self.parents):
            bone_lengths[joint_idx] = np.linalg.norm(self.offsets[joint_idx])
        return bone_lengths

    def get_joint_orientation(self):
        rotations = self.rotations.copy()
        rotations = rotations[..., [1, 2, 3, 0]] # wxyz -> xyzw
        
        global_joint_orientations = np.zeros((self.frame_num, self.joint_num, 4))
        global_joint_orientations[:, :, 3] = 1.0
        
        for joint_idx, parent_idx in enumerate(self.parents):
            parent_orientation = R.from_quat(global_joint_orientations[:, parent_idx, :])
            global_joint_orientations[:, joint_idx, :] = (parent_orientation * R.from_quat(rotations[:, joint_idx, :])).as_quat()
        
        return global_joint_orientations

    def get_joint_positions(self, no_root_motion=False):
        rotations = self.rotations.copy()
        rotations = rotations[..., [1, 2, 3, 0]] # wxyz -> xyzw
        
        global_joint_positions = np.zeros((self.frame_num, self.joint_num, 3))
        global_joint_orientations = np.zeros((self.frame_num, self.joint_num, 4))
        global_joint_orientations[:, :, 3] = 1.0
        
        for joint_idx, parent_idx in enumerate(self.parents):
            parent_orientation = R.from_quat(global_joint_orientations[:, parent_idx, :])
            rotated_vector = parent_orientation.apply(self.offsets[None, joint_idx, :])
            global_joint_positions[:, joint_idx, :] = global_joint_positions[:, parent_idx, :] + rotated_vector
            global_joint_orientations[:, joint_idx, :] = (parent_orientation * R.from_quat(rotations[:, joint_idx, :])).as_quat()
        
        if no_root_motion:
            return global_joint_positions
        else:
            global_joint_positions = global_joint_positions + self.positions[:, 0][:, np.newaxis, :]
            return global_joint_positions


    def rename_joints(self, old_clip, new_clip):
        for _idx, _name in enumerate(self.names):
            self.names[_idx] = _name.replace(old_clip, new_clip)
        
        old_keys = list(self.parent_names.keys())
        for _idx, _key in enumerate(old_keys):
            value = self.parent_names[_key]
            new_key = _key.replace(old_clip, new_clip)
            
            if value is not None:
                new_value = value.replace(old_clip, new_clip)
                self.parent_names[new_key] = new_value
            else:
                self.parent_names[new_key] = None
            self.parent_names.pop(_key)
        return self

    def get_joint_positions_with_end(self, local=False):
        # detect the end leaf in the patents
        end_leaf = []
        for joint_idx, parent_idx in enumerate(self.parents[:-1]):
            if self.parents[joint_idx+1] != joint_idx:
                end_leaf.append(joint_idx)
        end_leaf.append(len(self.parents)-1)
        assert len(end_leaf) == len(self.end_offsets)
        global_joint_positions = self.global_positions
        global_joint_orientations = self.get_joint_orientation()
        end_leaf_position = np.zeros((self.frame_num, len(end_leaf), 3))
        for idx, end_leaf_idx in enumerate(end_leaf):
            parent_orientation = R.from_quat(global_joint_orientations[:, end_leaf_idx, :])
            rotated_vector = parent_orientation.apply(self.end_offsets[None, idx, :])
            end_leaf_position[:, idx, :] = global_joint_positions[:, end_leaf_idx, :] + rotated_vector
        return end_leaf_position

    # get the local joint positions: relative to the root joint
    def get_local_joint_positions(self):
        global_joint_positions = self.global_positions
        local_joint_positions = global_joint_positions - global_joint_positions[:, 0][:, np.newaxis, :]
        return local_joint_positions

    def __getitem__(self, index):
        self.opt_history.append('slice')
        if isinstance(index, int):
            return_motion = self.copy()
            return_motion.rotations = return_motion.rotations[index:index+1]
            return_motion.positions = return_motion.positions[index:index+1]
            return_motion.frame_num = 1
            # return_motion.global_positions = return_motion.global_positions[index:index+1]
        elif isinstance(index, slice) or isinstance(index, np.ndarray) or isinstance(index, tuple) or isinstance(index, list):
            return_motion = self.copy()
            return_motion.rotations = return_motion.rotations[index]
            return_motion.positions = return_motion.positions[index]
            return_motion.frame_num = return_motion.rotations.shape[0]
            # return_motion._global_position = return_motion.global_positions[index]
        else:
            raise TypeError("Invalid argument type.")
        return return_motion

        
    def copy(self):
        return Motion(
            rotations=copy.deepcopy(self._rotations),
            positions=copy.deepcopy(self._positions),
            offsets=copy.deepcopy(self._offsets),
            parents=copy.deepcopy(self.parents),
            names=self.names[:],  
            frametime=self.frametime,
            _global_positions=copy.deepcopy(self._global_positions),
            end_offsets=copy.deepcopy(self.end_offsets),
            filepath=self.filepath, 
            sampling_factor=self.sampling_factor, 
            scaling_factor=self.scaling_factor, 
            removed_joints=self.removed_joints,
            opt_history=self.opt_history[:]
        )


    @classmethod
    def load_bvh(cls, filename, start=None, end=None, data_type=np.float32):
        
        if not os.path.exists(filename):
            raise Exception("File %s not found!" % filename)
        
        f = open(filename, "r")

        i = 0
        active = -1
        end_site = False
        orders = [] # for the case the the order is all same in the file
        
        names = []
        offsets = np.array([], dtype=data_type).reshape((0,3))
        end_offsets = np.array([], dtype=data_type).reshape((0,3))
        parents = np.array([], dtype=int)
        
        for line in f:
            
            if "HIERARCHY" in line: continue
            if "MOTION" in line: continue

            rmatch = re.match(r"ROOT (.+)", line)
            if rmatch:
                names.append(rmatch.group(1))
                offsets    = np.append(offsets,    np.array([[0,0,0]], dtype=data_type),   axis=0)
                parents    = np.append(parents, active)
                active = (len(parents)-1)
                continue

            if "{" in line: continue

            if "}" in line:
                if end_site: end_site = False
                else: active = parents[active]
                continue
            
            offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if offmatch:
                if not end_site:
                    offsets[active] = np.array([list(map(data_type, offmatch.groups()))])
                else:
                    end_offsets = np.append(end_offsets, np.array([list(map(data_type, offmatch.groups()))]), axis=0)
                continue
            
            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2+channelis:2+channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
                orders.append(order)
                continue

            jmatch = re.match(r"\s*JOINT (.+)", line)
            if jmatch:
                names.append(jmatch.group(1))
                offsets    = np.append(offsets,    np.array([[0,0,0]], dtype=data_type),   axis=0)
                parents    = np.append(parents, active)
                active = (len(parents)-1)
                continue
            
            if "End Site" in line or "End site" in line:
                end_site = True
                continue
                
            fmatch = re.match("\s*Frames:\s+(\d+)", line)
            if fmatch:
                if start and end:
                    fnum = (end - start)-1
                else:
                    fnum = int(fmatch.group(1))
                jnum = len(parents)
                # result: [fnum, J, 3]
                positions = offsets[np.newaxis].repeat(fnum, axis=0)
                # result: [fnum, len(orients), 3]
                rotations = np.zeros((fnum, jnum, 3), dtype=data_type)
                continue
            
            fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            if fmatch:
                frametime = float(fmatch.group(1))
                continue
            
            if (start and end) and (i < start or i >= end-1):
                i += 1
                continue
            
            dmatch = line.strip().split()
            if dmatch:
                data_block = np.array(list(map(float, dmatch)))
                N = len(parents)
                fi = i - start if start else i
                if channels == 3:
                    # This should be root positions[0:1] & all rotations
                    positions[fi,0:1] = data_block[0:3]
                    rotations[fi, : ] = data_block[3: ].reshape(N,3)
                elif channels == 6:
                    data_block = data_block.reshape(N,6)
                    # fill in all positions
                    positions[fi,:] = data_block[:,0:3]
                    rotations[fi,:] = data_block[:,3:6]
                elif channels == 9:
                    positions[fi,0] = data_block[0:3]
                    data_block = data_block[3:].reshape(N-1,9)
                    rotations[fi,1:] = data_block[:,3:6]
                    positions[fi,1:] += data_block[:,0:3] * data_block[:,6:9]
                else:
                    raise Exception("Too many channels! %i" % channels)
                i += 1

        f.close()
        orders = [order.upper() for order in orders]
        rotations = np.radians(rotations)
        
        rotation_eular, rotation_q = [], []
        for joint_idx in range(rotations.shape[1]):
            order = orders[joint_idx]
            R_j = R.from_euler(seq=orders[joint_idx], angles=rotations[:, joint_idx])
            rotation_eular.append(R_j.as_euler('XYZ'))
            rotation_q.append(R_j.as_quat()[..., [3, 0, 1, 2]]) #xyzw -> wxyz
        
        rotation_eular, rotation_q = np.array(rotation_eular, dtype=data_type).transpose((1, 0, 2)), clean_quat(np.array(rotation_q, dtype=data_type).transpose((1, 0, 2)))
        
        return cls(rotation_q, positions, offsets, parents, names, frametime, end_offsets, filepath=filename)


    def export(self, filename, order='XYZ', save_positions=False, save_reset=False, save_root_position=True, save_ori_scal=True, force_export=False):
        
        self.end_site_count = 0
        
        def save_joint(f, rotations, offsets, parents, names, t, i, order, save_positions=False):
            global end_site_count
            f.write("%sJOINT %s\n" % (t, names[i]))
            f.write("%s{\n" % t)
            t += '\t'
        
            f.write("%sOFFSET %f %f %f\n" % (t, offsets[i,0], offsets[i,1], offsets[i,2]))
            
            if save_positions:
                f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t, 
                    channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
            else:
                f.write("%sCHANNELS 3 %s %s %s\n" % (t, 
                    channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
            
            end_site = True
            
            for j in range(rotations.shape[1]):
                if parents[j] == i:
                    t = save_joint(f, rotations, offsets, parents, names, t, j, order=order, save_positions=save_positions)
                    end_site = False
            
            if end_site:
                f.write("%sEnd Site\n" % t)
                f.write("%s{\n" % t)
                t += '\t'
                f.write("%sOFFSET %f %f %f\n" % (t, self.end_offsets[self.end_site_count, 0], self.end_offsets[self.end_site_count, 1], self.end_offsets[self.end_site_count, 2]))
                t = t[:-1]
                f.write("%s}\n" % t)
                self.end_site_count += 1
            t = t[:-1]
            f.write("%s}\n" % t)
            return t
        
        # Start of export function
        if force_export:
            file_folder = os.path.dirname(filename)
            if not os.path.exists(file_folder):
                os.makedirs(file_folder, exist_ok=True)
        
        if self.names is None:
            self.names = ["joint_" + str(i) for i in range(len(self.parents))]
        
        order = order.upper()
        
        rots = np.zeros((self.frame_num, self.joint_num, 3))
        if save_ori_scal:
            ori_offset = self.offsets / self.scaling_factor
            self.end_offsets = self.end_offsets / self.scaling_factor
            ori_positions = self.positions / self.scaling_factor
        else:
            ori_offset = self.offsets
            ori_positions = self.positions
            
        for joint_idx in range(self.joint_num):
            joint_rot = self.rotations[:, joint_idx]
            if joint_rot.shape[-1] == 4:
                joint_rot = joint_rot[..., [1, 2, 3, 0]] # wxyz -> xyzw
                rots[:, joint_idx] = R.from_quat(joint_rot).as_euler(order, degrees=True)
            else:
                rots[:, joint_idx] = R.from_euler(seq=order, angles=joint_rot).as_euler(order, degrees=True)
            print
        
        # convert from radians to degrees
        with open(filename, 'w') as f:

            t = ""
            f.write("%sHIERARCHY\n" % t)
            f.write("%sROOT %s\n" % (t, self.names[0]))
            f.write("%s{\n" % t)
            t += '\t'

            f.write("%sOFFSET %f %f %f\n" % (t, ori_offset[0,0], ori_offset[0,1], ori_offset[0,2]) )
            f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % 
                (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

            for i in range(self.joint_num):
                if self.parents[i] == 0:
                    t = save_joint(f, self.rotations, ori_offset, self.parents, self.names, t, i, order=order, save_positions=save_positions)

            t = t[:-1]
            f.write("%s}\n" % t)

            f.write("MOTION\n")
            f.write("Frames: %i\n" % self.frame_num)
            f.write("Frame Time: %f\n" % self.frametime)
            
            poss = ori_positions
            
            for i in range(self.frame_num):
                for j in range(self.joint_num):
                    if i == 0 and save_reset:
                        if j == 0:
                            f.write("%f %f %f %f %f %f " % (0, 0, 0, 0, 0, 0))
                        else:
                            f.write("%f %f %f " % (0, 0, 0))
                    else:
                        if save_positions or j == 0:
                            if save_root_position:
                                f.write("%f %f %f %f %f %f " % (poss[i,j,0], poss[i,j,1], poss[i,j,2], rots[i,j,0], rots[i,j,1], rots[i,j,2]))
                            else:
                                f.write("%f %f %f %f %f %f " % (0, 0, 0, rots[i,j,0], rots[i,j,1], rots[i,j,2]))
                        else:
                            f.write("%f %f %f " % (rots[i,j,0], rots[i,j,1], rots[i,j,2]))
                f.write("\n")

    def plot(self, save_path, save_frame=0):
        import matplotlib.pyplot as plt
        positions = self.global_positions
        positions_2d = positions[:, :, [2, 1]]
        plt.figure(figsize=(10, 10))
        plt.scatter(positions_2d[save_frame, :, 0], positions_2d[save_frame, :, 1])
        
        for joint_idx, parent_idx in enumerate(self.parents):
            if parent_idx == -1:
                continue
            plt.plot(positions_2d[save_frame, [joint_idx, parent_idx], 0], positions_2d[save_frame, [joint_idx, parent_idx], 1])
        
        plt.savefig(save_path)
        plt.close()