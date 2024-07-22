# Evaluation 

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