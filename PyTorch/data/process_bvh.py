'''
Target: 
1. Remove the unused joints
2. Make the root position to be (0, 0, 0)
3. Align the root forwarding to x-axis
'''

import sys
sys.path.append('./')

import os
import utils.motion_modules as motion_modules
import style_helper as style100

from multiprocessing import Pool
from utils.bvh_motion import Motion

def process_bvh(bvh_path, output_path):
    raw_motion = Motion.load_bvh(bvh_path)
    raw_motion.offsets[0].fill(0)
    motion = motion_modules.remove_joints(raw_motion, ['eye', 'index', 'middle', 'ring', 'pinky', 'thumb'])
    motion = motion_modules.root(motion)
    motion = motion_modules.temporal_scale(motion, 2)
    forward_angle = motion_modules.extract_forward(motion, 0, style100.left_shoulder_name, style100.right_shoulder_name, style100.left_hip_name, style100.right_hip_name)
    motion = motion_modules.rotate(motion, given_angle=forward_angle, axis='y')
    motion = motion_modules.on_ground(motion)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    motion.export(output_path)
    print('Finish processing %s' % bvh_path)


if __name__ == '__main__':
    
    process_folder = 'data/100STYLE_mixamo/raw'
    output_folder = 'data/100STYLE_mixamo/simple'

    bvh_paths = []
    for root, dirs, files in os.walk(process_folder):
        for file in files:
            if file.endswith('.bvh'):
                bvh_paths.append(os.path.join(root, file))
                
    output_paths = [f.replace(process_folder, output_folder) for f in bvh_paths]
    
    print('Total BVH files: %d' % len(bvh_paths))
    pool = Pool(processes=os.cpu_count()) 
        
    tasks = []
    for idx in range(len(bvh_paths)):
        bvh_path = bvh_paths[idx]
        output_path = output_paths[idx]
        pool.apply_async(process_bvh, args=(bvh_path, output_path))

    pool.close()  
    pool.join() 
