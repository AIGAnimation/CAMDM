import os
import numpy as np
from tqdm import tqdm
import scipy

def calculate_metrics(args):
    bvh_path, gt_distribution, save_list = args
    feature_path = bvh_path.replace('.bvh', '_predicted.npz')
    feature = np.load(feature_path)['feats']
    if 'exp1' in bvh_path or 'diffusion_step' in bvh_path:
        style_name = bvh_path.split('/')[-1].split('_')[-1].split('.')[0]
        gt_mu, gt_sigma = gt_distribution[style_name]
        mu, sigma = np.mean(feature, axis=0), np.cov(feature, rowvar=False)
        ssd = np.sum((mu - gt_mu)**2.0)
        covmean = scipy.linalg.sqrtm(sigma.dot(gt_sigma))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = ssd + np.trace(sigma + gt_sigma - 2.0 * covmean)
        save_list.append(fid/len(sigma))
    if 'exp2' in bvh_path:
        start_style_name = bvh_path.split('/')[-1].split('.')[0].split('_')[1]
        gt_mu, gt_sigma = gt_distribution[start_style_name]
        start_feature = feature[:len(feature)//2]
        mu, sigma = np.mean(start_feature, axis=0), np.cov(start_feature, rowvar=False)
        ssd = np.sum((mu - gt_mu)**2.0)
        covmean = scipy.linalg.sqrtm(sigma.dot(gt_sigma))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        start_fid = ssd + np.trace(sigma + gt_sigma - 2.0 * covmean)
        
        target_style_name = bvh_path.split('/')[-1].split('.')[0].split('_')[3]
        gt_mu, gt_sigma = gt_distribution[target_style_name]
        target_feature = feature[len(feature)//2:]
        mu, sigma = np.mean(target_feature, axis=0), np.cov(target_feature, rowvar=False)
        ssd = np.sum((mu - gt_mu)**2.0)
        covmean = scipy.linalg.sqrtm(sigma.dot(gt_sigma))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        target_fid = ssd + np.trace(sigma + gt_sigma - 2.0 * covmean)
        save_list.append((start_fid + target_fid)/2/len(sigma))
    # print('Finish %s' % bvh_path)

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
        "data/exp2_ours"
    ]

    result_path = os.path.join('./result_recording', 'motion_FID.txt')
    gt_distribution = np.load('feature_extractor/100style_position_distribution.npz', allow_pickle=True)['style_features_dic'].item()

    if not os.path.exists(result_path):
        with open(result_path, 'w') as f:
            f.write('Metrics: \t\t FID\t\n')
            f.write('---------------------------------\n')

    for test_folder in test_folders:
        test_file_list = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.bvh')]
        calculate_metrics((test_file_list[0], gt_distribution, []))

        metric_list = []
        args = [(test_file, gt_distribution, metric_list) for test_file in test_file_list]
        for arg in tqdm(args):
            calculate_metrics(arg)
        
        
        # num_processes = multiprocessing.cpu_count()
        # with multiprocessing.Pool(processes=num_processes) as pool:
        #     pool.map(calculate_metrics, args)
        
        # metric_list = list(metric_list)
        avg_value = np.mean(metric_list)
        
        with open(result_path, 'a') as f:
            f.write(test_folder + '\t' + '%.4f\t\n' % avg_value)
            f.write('\n')

        print('Finish %s, metrics: %s' % (test_folder, avg_value))