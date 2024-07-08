import os
import numpy as np
from tqdm import tqdm
import utils.motion_modules as motion_modules
from utils.bvh_motion import Motion
import multiprocessing


def calculate_metrics(args):
    bvh_path, metric_names, save_dic = args
    motion = Motion.load_bvh(bvh_path)
    
    foot_idx = []
    if 'RightToeBase' in motion.names:
        foot_idx.append(motion.names.index('LeftToeBase'))
        foot_idx.append(motion.names.index('RightToeBase'))
    if 'RightToe' in motion.names:
        foot_idx.append(motion.names.index('LeftToe'))
        foot_idx.append(motion.names.index('RightToe'))    
    motion = motion_modules.on_ground(motion, foot_idx)
    metric_value = []
    for metric_name in metric_names:
        if metric_name == 'foot_sliding':
            metric_value.append(motion_modules.cal_metric_foot_sliding(motion, y_threshold=1)) # cm
        elif metric_name == 'cohenrence':
            metric_value.append(motion_modules.cal_metric_cohenrence(motion))
    save_dic[bvh_path] = metric_value
    

if __name__ == '__main__':

    test_folders = [
        "data/exp1_mann+dp",
        "data/exp1_mann+lp",
        "data/exp1_moglow",
        "data/exp1_ours",
        "data/exp1_ours_mlp",
        "data/exp2_mann+dp",
        "data/exp2_mann+lp",
        "data/exp2_moglow",
        "data/exp2_ours",
    ]

    select_metric = ['foot_sliding', 'cohenrence']
    result_path = os.path.join('./result_recording', 'motion_quality.txt')

    if not os.path.exists(result_path):
        with open(result_path, 'w') as f:
            f.write('Metrics: \t\t')
            for metric_name in select_metric:
                f.write('%s\t\t' % metric_name)
            f.write('\n')
            f.write('---------------------------------\n')

    for test_folder in test_folders:
        
        test_file_list = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.bvh')]

        calculate_metrics((test_file_list[0], select_metric, {}))

        metric_dic = multiprocessing.Manager().dict()
        args = [(test_file, select_metric, metric_dic) for test_file in test_file_list]
        
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