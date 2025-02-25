# Evaluation 

## Fair comparison settings

Our system consists of two modules: AR model training in PyTorch and the runtime part in Unity. It would not be fair if we simply output the motion using Python without the runtime strategies. However, re-implementing all these features in Python is non-trivial.
Hence, we provide the following features in Unity to make a fair comparison:
1. Load predefined control signals for all styles.
2. Export the generated motion data in Unity runtime.

You can check the the main Unity inspector, which should contain the following feature in the 'Evaluation' component:

If the 'Load Control' is checked, the control signals will be loaded from the predefined file in the 'Load Control Path'. If the 'Export Motion' is checked, the motion data will be exported to the 'Export BVH Path' in BVH format after the motion generation.

WARNING: The current version of BVH header is only compatible with our character model. If you want to use it with other characters, you need to modify the BVH header manually by simply copying the hierarchy part from the original BVH file.

<p align="center">
<img src="https://github.com/user-attachments/assets/cd595cbe-4e25-46a2-abec-d5255329cb09" width="400">
</p>


## Data and pre-trained encoders

We prepare all the comparison data with this [Onedrive](https://1drv.ms/u/s!AtagzSrg3_hppdxIvlkNf35Jxf-PBw?e=6cJV5H) or [BaiduDisk](https://pan.baidu.com/s/1Tvc93P_OLyrntQyHJVHorw?pwd=62pf) link. You can download it and prepare this folder with following structure:

```
data\
---- exp1_mann+dp
---- exp1_mann+lp
---- exp1_moglow
---- exp1_ours
...
feature_extractor\
---- 100style_position_distribution.npz
---- 100style_position_classifier.pth
...
utils\
...
result_recording\
...
```

## Commands

Then you can run our evaluation with following commands:
``` shell
# Produce the deep latent for each motion data
python compute_feat.py

# Compute the metrics
python eval_motion_FID.py
python eval_motion_quality.py
python eval_traj_alignment.py
```