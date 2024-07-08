import os
import pickle
import numpy as np
from tqdm import tqdm

import math
import multiprocessing
from multiprocessing import Pool
import utils.motion_modules as motion_modules
from utils.bvh_motion import Motion


target_fps = 30
styleList = ["Aeroplane", "Akimbo", "Angry", "ArmsAboveHead", "ArmsBehindBack", "ArmsBySide", "ArmsFolded", "Balance", "BeatChest", 
			"BentForward", "BentKnees", "BigSteps", "BouncyLeft", "BouncyRight", "Cat", "Chicken", "CrossOver", "Crouched", "CrowdAvoidance", 
			"Depressed", "Dinosaur", "DragLeftLeg", "DragRightLeg", "Drunk", "DuckFoot", "Elated", "FairySteps", "Flapping", "FlickLegs", "Followed",  
			"GracefulArms", "HandsBetweenLegs", "HandsInPockets", "Heavyset", "HighKnees", "InTheDark", "KarateChop", "Kick", "LawnMower", "LeanBack", "LeanLeft", 
			"LeanRight", "LeftHop", "LegsApart", "LimpLeft", "LimpRight", "LookUp", "Lunge", "March", "Monk", "Morris", "Neutral", "Old", "OnHeels", "OnPhoneLeft", 
			"OnPhoneRight", "OnToesBentForward", "OnToesCrouched", "PendulumHands", "Penguin", "PigeonToed", "Proud", "Punch", "Quail", "RaisedLeftArm", "RaisedRightArm", 
			"RightHop", "Roadrunner", "Robot", "Rocket", "Rushed", "ShieldedLeft", "ShieldedRight", "Skip", "SlideFeet", "SpinAntiClock", "SpinClock", "Star", "StartStop", 
			"Stiff", "Strutting", "Superman", "Swat", "Sweep", "Swimming", "SwingArmsRound", "SwingShoulders", "Teapot", "Tiptoe", "TogetherStep", "TwoFootJump", 
			"WalkingStickLeft", "WalkingStickRight", "Waving", "WhirlArms", "WideLegs", "WiggleHips", "WildArms", "WildLegs", "Zombie"]


def process_file(bvh_paths):
    bvh_path, skeleton_config = bvh_paths
    if '.bvh' not in bvh_path:
        return None

    motion = Motion.load_bvh(bvh_path)
    motion = motion_modules.scaling(motion, 0.01)
    motion = motion_modules.root(motion)
    
    fps = math.ceil(1 / motion.frametime)
    temporal_scale = int(fps / target_fps)
    temporal_scale = max(temporal_scale, 1)
    
    motion = motion_modules.temporal_scale(motion, temporal_scale)
    
    all_frame_idx = np.arange(motion.frame_num)
    
    if skeleton_config == 'style_skeleton':
        forward_angle = motion_modules.extract_forward(motion, all_frame_idx, 'LeftCollar', 'RightCollar', 'LeftHip', 'RightHip')
    elif skeleton_config == 'ybot_skeleton':
        forward_angle = motion_modules.extract_forward(motion, all_frame_idx, 'LeftShoulder', 'RightShoulder', 'LeftUpLeg', 'RightUpLeg')
    
    motion = motion_modules.rotate(motion, given_angle=forward_angle, axis='y')
    motion = motion_modules.on_ground(motion)
    
    global_position = motion.update_global_positions()
    local_position = global_position - global_position[:, 0:1, :]
    
    if skeleton_config == 'style_skeleton':
        return_joint_names = ['Hips', 'Neck', 'Head', 'RightCollar', 'RightShoulder', 'RightElbow', 'LeftCollar', 'LeftShoulder', 'LeftElbow', 'RightHip', 'RightKnee', 'RightAnkle', 'LeftHip', 'LeftKnee', 'LeftAnkle']
    elif skeleton_config == 'ybot_skeleton':
        return_joint_names = ['Hips', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot']
    
    return_joint_idx = [motion.names.index(joint_name) for joint_name in return_joint_names]
    local_position = local_position[:, return_joint_idx, :]
    
    style_name = bvh_path.split('/')[-1].split('_')[0]
    style_idx = styleList.index(style_name) if style_name in styleList else -1
    
    return {'local_position': local_position, 'style_name': style_name, 'style_idx': style_idx, 'file_path': bvh_path}


def make_data(bvh_paths, save_path):
    
    # data = process_file(bvh_paths[0])
    
    with Pool(multiprocessing.cpu_count()) as p:
        data_list = list(tqdm(p.imap(process_file, bvh_paths), total=len(bvh_paths)))

    data_list = [data for data in data_list if data is not None]
    pickle.dump(data_list, open(save_path, 'wb'))
    

if __name__ == '__main__':
    
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
        
    make_data(bvh_paths, 'data/100style_position_ori+ybot.pkl')
    
    print('Done!')