 # <p align="center"> Taming Diffusion Probabilistic Models for Character Control </p>

 #####  <p align="center"> [Rui Chen*](https://aruichen.github.io/), [Mingyi Shi*](https://rubbly.cn/), [Shaoli Huang](https://scholar.google.com/citations?user=o31BPFsAAAAJ&hl=en), [Ping Tan](https://ece.hkust.edu.hk/pingtan), [Taku Komura](https://scholar.google.com.hk/citations?user=TApLOhkAAAAJ&hl=en), [Xuelin Chen](https://xuelin-chen.github.io/)</p>
 ##### <p align="center"> SIGGRAPH 2024
 ##### <p align="center"> *equal contribution
 
#### <p align="center">[ArXiv](https://arxiv.org/abs/2404.15121) | [Project Page](https://aiganimation.github.io/CAMDM/) | [Video](https://www.youtube.com/watch?v=J9L0fR_x5OA) | [Unity demo](https://drive.google.com/file/d/1NYXP-fbEegErfaIgtHXvvrrfLXUSqYXg/view?usp=sharing)</p>

<p align="center">
  <img width="40%" src="assets/camdm.png"/>
</p>

# Update log
- (2024.04.24) 
  - Release the windows Unity demo (GPU) trained in 100style dataset.
- (2024.06.23) 
  - Release the training code in pytorch.

## Getting Started

This code has been tested on `Ubuntu 20.04.6 LTS` and requires the following:

* Python 3.8 and PyTorch 1.13.1 (for exporting Unity-compatible ONNX models)
* Anaconda (conda3) or Miniconda3
* A CUDA-capable GPU (a single GPU is sufficient)
* It takes about 14 hours to get the best performance of the netwoek in a single 3090 GPU.
### 1. Environment Setup

Create and activate the conda environment using the provided `environment.yml` file:
```shell
git clone https://github.com/AIGAnimation/CAMDM.git
cd CAMDM/training_code
conda env create -f environment.yml
conda activate cdmdm
```

### 2. Data Preparation

Download the retargeted 100style dataset using this [OneDrive link](https://1drv.ms/u/s!AtagzSrg3_hppOVH-uNQCPXgwKN9Rg?e=wQH2aT), and then move it into the `data` folder of this project:

```shell
mv 100STYLE_mixamo.zip ./data
mkdir -p data/pkls
unzip data/100STYLE_mixamo.zip -d ./data/100STYLE_mixamo
```

To expedite training, we have omitted certain joints from the original Mixamo skeleton, such as those of the fingers, which are also not present in the original 100style dataset. Subsequently, we package the data into a `.pkl` file:

```shell
python data/process_bvh.py
python data/make_pose_data.py
```

### Network Training

To train the network with the default settings, use the following command:

```shell
python train.py --cluster -n gen_step4 --epoch 500 --batch_size 512 --diffusion_steps 4  # The checkpoint of approximately the 50th epoch can be exported
```

We offer various options for network training. For a detailed explanation, please refer to our [option.py](training_code/config/option.py) and [default.json](training_code/config/default.json) files. For instance, the `--cluster` option facilitates training in a cluster environment by allowing the training to resume automatically if it is interrupted.

### ONNX Export

After training, you can select the best checkpoint and export it as an ONNX file, which can be imported into Unity. The export path will be set to the same directory as the checkpoint:

```shell
# Replace 'CHECKPOINT_PATH' with the path to your actual checkpoint file
python export_onnx.py --checkpoint CHECKPOINT_PATH

```

## Todo
- [x] Release unity .exe demo in windows. （2024.04.24）
- [x] Release the training code in pytorch. （2024.06.23）
- [ ] Release the inference code in unity. （will release before 06.26）
- [ ] Release the evaluation code in paper. （will release before 06.30）

## BibTex
```
@inproceedings{camdm,
  title={Taming Diffusion Probabilistic Models for Character Control},
  author={Rui Chen and Mingyi Shi and Shaoli Huang and Ping Tan and Taku Komura and Xuelin Chen},
  booktitle={SIGGRAPH},
  year={2024}
}
```
