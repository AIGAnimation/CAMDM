import os
import csv
import copy

left_foot_name = 'mixamorig:LeftFoot'
right_foot_name = 'mixamorig:RightFoot'
left_shoulder_name = 'mixamorig:LeftShoulder'
right_shoulder_name = 'mixamorig:RightShoulder'
left_hip_name = 'mixamorig:LeftUpLeg'
right_hip_name = 'mixamorig:RightUpLeg'


def get_info(root_folder_path, meta_file='Dataset_List.csv', framecuts_file='Frame_Cuts.csv'):
    
    style_info_dic = {}
    style_item_template = {
        'description': '',
        'is_stochastic': False,
        'is_symmetric': False,
        'notes': '',
        'framecuts': {
            'BR': [0, 0], 'BW': [0, 0], 'FR': [0, 0], 'FW': [0, 0], 'ID': [0, 0], 
            'SR': [0, 0], 'SW': [0, 0], 'TR1': [0, 0], 'TR2': [0, 0], 'TR3': [0, 0],
        }
    }
    
    # load meta information
    with open(os.path.join(root_folder_path, meta_file), 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx > 0:
                style_name = row[0]
                style_item = copy.deepcopy(style_item_template)
                style_item['description'], style_item['notes'] = row[1], row[4]
                if 'y' in row[2].lower():
                    style_item['is_stochastic'] = True
                if 'y' in row[3].lower():
                    style_item['is_symmetric'] = True
                style_item['notes'] = row[4]
                style_info_dic[style_name] = style_item
    
    def get_frame_num(number: str):
        if number == 'N/A' or 'n' in number.lower():
            return None
        return int(number)
    
    # load frame cut information
    with open(os.path.join(root_folder_path, framecuts_file), 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx > 0:
                style_name = row[0]
                if style_name not in style_info_dic.keys():
                    continue
                style_info_dic[style_name]['framecuts']['BR'] = [get_frame_num(row[1]), get_frame_num(row[2])]
                style_info_dic[style_name]['framecuts']['BW'] = [get_frame_num(row[3]), get_frame_num(row[4])]
                style_info_dic[style_name]['framecuts']['FR'] = [get_frame_num(row[5]), get_frame_num(row[6])]
                style_info_dic[style_name]['framecuts']['FW'] = [get_frame_num(row[7]), get_frame_num(row[8])]
                style_info_dic[style_name]['framecuts']['ID'] = [get_frame_num(row[9]), get_frame_num(row[10])]
                style_info_dic[style_name]['framecuts']['SR'] = [get_frame_num(row[11]), get_frame_num(row[12])]
                style_info_dic[style_name]['framecuts']['SW'] = [get_frame_num(row[13]), get_frame_num(row[14])]
                style_info_dic[style_name]['framecuts']['TR1'] = [get_frame_num(row[15]), get_frame_num(row[16])]
                style_info_dic[style_name]['framecuts']['TR2'] = [get_frame_num(row[17]), get_frame_num(row[18])]
                style_info_dic[style_name]['framecuts']['TR3'] = [get_frame_num(row[19]), get_frame_num(row[20])]
    
    return style_info_dic