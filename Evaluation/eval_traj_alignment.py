import os
import numpy as np
from tqdm import tqdm
import utils.motion_modules as motion_modules
from utils.bvh_motion import Motion
import multiprocessing


def calculate_metrics(args):
    bvh_path, preset_path, metric_names, save_dic = args
    motion = Motion.load_bvh(bvh_path)
    
    all_frame_idx = np.arange(motion.frame_num)
    if 'LeftCollar' in motion.names:
        forward_angle = motion_modules.extract_forward(motion, all_frame_idx, 'LeftCollar', 'RightCollar', 'LeftHip', 'RightHip')
    elif 'RightUpLeg' in motion.names:
        forward_angle = motion_modules.extract_forward(motion, all_frame_idx, 'LeftShoulder', 'RightShoulder', 'LeftUpLeg', 'RightUpLeg')

    forward_angle = np.rad2deg(forward_angle)
    preset_traj, preset_orien = [], []
    with open(preset_path, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            traj_x, traj_z = float(values[0]), float(values[2])
            traj_angle = np.arctan2(traj_z, traj_x)
            dirc_x, dirc_z = float(values[3]), float(values[5])
            dirc_angle = np.arctan2(dirc_z, -dirc_x)
            preset_traj.append(traj_angle)
            preset_orien.append(dirc_angle)
    preset_traj, preset_orien = np.rad2deg(np.array(preset_traj)), np.rad2deg(np.array(preset_orien))
    
    if motion.frame_num > len(preset_traj):
        motion = motion_modules.temporal_scale(motion, 2)
        forward_angle = forward_angle[::2]
    
    preset_traj = preset_traj[:motion.frame_num-1]
    preset_orien = preset_orien[:motion.frame_num]
    
    traj_pos = motion.positions[:, 0, [0, 2]]
    traj_pos[:, 0] *= -1
    traj_pos_diff = np.diff(traj_pos, axis=0)
    traj_angle = np.rad2deg(np.arctan2(traj_pos_diff[:, 1], traj_pos_diff[:, 0])) 
    
    metric_value = []
    for metric_name in metric_names:
        if metric_name == 'traj_error':
            value = np.abs(np.mean(preset_traj - traj_angle))
            # value = np.min([value, 90 - value, 180 - value])
            metric_value.append(np.abs(value))
        elif metric_name == 'orien_error':
            value = np.abs(np.mean(preset_orien - forward_angle))
            # value = np.min([value, 90 - value, 180 - value])
            metric_value.append(np.abs(value))
    
    save_dic[bvh_path] = metric_value
    

if __name__ == '__main__':

    test_folders = [
            ["data/exp1_mann+dp", "data/preset.txt"],
            ["data/exp1_mann+lp", "data/preset.txt"],
            ["data/exp1_matching", "data/preset.txt"],
            ["data/exp1_moglow", "data/preset.txt"],
            ["data/exp1_ours", "data/preset.txt"],
            ["data/exp1_ours_mlp", "data/preset.txt"],
        ]

    select_metric = ['traj_error', 'orien_error']
    result_path = os.path.join('./result_recording', 'trajectory_alignment.txt')

    if not os.path.exists(result_path):
        with open(result_path, 'w') as f:
            f.write('Metrics: \t\t')
            for metric_name in select_metric:
                f.write('%s\t\t' % metric_name)
            f.write('\n')
            f.write('---------------------------------\n')

    for test_folder, preset_path in test_folders:
        test_file_list = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.bvh')]

        calculate_metrics((test_file_list[0], preset_path, select_metric, {}))

        metric_dic = multiprocessing.Manager().dict()
        args = [(test_file, preset_path, select_metric, metric_dic) for test_file in test_file_list]
        
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(calculate_metrics, args)
        
        metric_dic = dict(metric_dic)

        avg_list = []
        for metric_name in select_metric:
            metric_value = []
            for test_file in test_file_list:
                metric_value.append(metric_dic[test_file][select_metric.index(metric_name)])
            # remove nan in metric_value
            metric_value = [value for value in metric_value if not np.isnan(value)]
            avg_list.append(np.mean(metric_value))
        
        with open(result_path, 'a') as f:
            f.write(test_folder + '\t')            
            for metric_name in select_metric:
                f.write('%.4f\t\t' % avg_list[select_metric.index(metric_name)])
            f.write('\n')

        print('Finish %s, metrics: %s' % (test_folder, avg_list))
            
            
            