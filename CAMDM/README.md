# Network Training

This code has been tested on `Ubuntu 20.04.6 LTS` and requires the following:

* Python 3.8 and PyTorch 1.13.1 (for exporting Unity-compatible ONNX models)
* Anaconda (conda3) or Miniconda3
* A CUDA-capable GPU (a single GPU is sufficient)
* It takes about 14 hours to get the best performance of the netwoek in a single 3090 GPU.

## Prerequisites

### 1. Environment Setup

Create and activate the conda environment using the provided `environment.yml` file:

```shell
git clone https://github.com/AIGAnimation/CAMDM.git
cd CAMDM/PyTorch
conda env create -f environment.yml
conda activate camdm
```

### 2. Data Download

You can download the raw 100style dataset and then use our script ([ARP-Batch-Retargeting](https://github.com/Shimingyi/ARP-Batch-Retargeting)) for reproducing the retargeting data. Or for your convenience, you can also download the retargeted 100style dataset using this [OneDrive link](https://1drv.ms/u/s!AtagzSrg3_hppdhGhO7hO6SfJx20LA?e=VGOE5E) or [BaiduDisk](https://pan.baidu.com/s/1jwymYGZ40DAryDz1kIDgXg?pwd=gwj6), and then move it into the `data` folder of this project:

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

Expected folder structure after processing:

```
data/
---- 100STYLE_mixamo/
--------- raw/
--------- simple/
--------- Dataset_List.csv
--------- Frame_Cuts.csv
---- pkls/
--------- 100style.pkl
```

## Training

To train the network with the default settings, use the following command:

```shell
python train.py -n camdm --epoch 100 --batch_size 512 --diffusion_steps 4 
```

We offer various options for network training. For a detailed explanation, please refer to our [option.py](config/option.py) and [default.json](config/default.json) files. For instance, the `--cluster` option facilitates training in a cluster environment by allowing the training to resume automatically if it is interrupted.

More parameters have been detailed below:

```shell
--rot_req: Rotation representation: choose from "q", "euler", or "6d".
--loss_terms: Loss terms to use in training. Format: [mse_rotation, positional_loss, velocity_loss, foot_contact]. Use 0 for No, 1 for Yes, e.g., "1111" in default.
--latent_dim: Width of the Transformer/GRU layers. 1024 in default.
--diffusion_steps: Number of diffusion steps. 4 steps can yield acceptable performance with M1 Macbook. 
```

### Training visualization

We use tensorboard to visualize the losses when training. In the training, you can go into checkpoint folder and run this command:

```
tensorboard --logdir=./
```

Then you can visit this link in your machine http://localhost:6006/ to check the visualzation.

<img width="1660" alt="losses" src="https://github.com/AIGAnimation/CAMDM/assets/7709951/510846ba-512b-4503-9aae-86312e69f344">

### Training convergence

Training for 100 epochs on a single RTX 3090 graphics card will take approximately one day. Extending the training beyond this point offers minimal improvement in loss reduction and may negatively impact output diversity.

Checkpoints will be saved every 10 epochs. Based on our experiments, you can begin evaluating performance after 20 epochs. There are few ways to check the training performance: 1. Check the exported bvh samples in the save folder; 2. Export the checkpoint file to ONNX and load it in our Unity program(More recommended).

## ONNX Export

After training, you can select the checkpoint and export it as an ONNX file, which can be imported into Unity. The export path will be set to the same directory as the checkpoint:

```shell
# Replace 'CHECKPOINT_PATH' with the path to your actual checkpoint file
python export_onnx.py -r CHECKPOINT_PATH
```

And it will produce a folder contains thress files which are used in Unity:

```
camdm_100style.onnx: Store the network weights
camdm_100style.json: Store the meta information about the character and diffusions
100style_conditions.json: Store all conditions
```

## Acknowledgement

Some of the code originates from the following projects. Please consider citing their work:

- [Guided-Diffusion](https://github.com/openai/guided-diffusion)
- [MDM](https://github.com/GuyTevet/motion-diffusion-model)
- [PFNN](https://github.com/sebastianstarke/AI4Animation)

## BibTex

```
@inproceedings{camdm,
  title={Taming Diffusion Probabilistic Models for Character Control},
  author={Rui Chen and Mingyi Shi and Shaoli Huang and Ping Tan and Taku Komura and Xuelin Chen},
  booktitle={SIGGRAPH},
  year={2024}
}
```
