# <p align="center"> Taming Diffusion Probabilistic Models for Character Control </p>

##### <p align="center"> [Rui Chen*](https://aruichen.github.io/), [Mingyi Shi*](https://rubbly.cn/), [Shaoli Huang](https://scholar.google.com/citations?user=o31BPFsAAAAJ&hl=en), [Ping Tan](https://ece.hkust.edu.hk/pingtan), [Taku Komura](https://scholar.google.com.hk/citations?user=TApLOhkAAAAJ&hl=en), [Xuelin Chen](https://xuelin-chen.github.io/) </p>

##### <p align="center"> SIGGRAPH 2024

##### <p align="center"> *equal contribution

<!-- #### <p align="center">[ArXiv](https://arxiv.org/abs/2404.15121) | [Project Page](https://aiganimation.github.io/CAMDM/) | [Video](https://www.youtube.com/watch?v=J9L0fR_x5OA) | [Unity demo](https://drive.google.com/file/d/1NYXP-fbEegErfaIgtHXvvrrfLXUSqYXg/view?usp=sharing)</p> -->

<p align="center">
  <img width="40%" src="https://github.com/AIGAnimation/CAMDM/assets/7709951/645d9882-8d13-48f4-9d54-be06acbf8c3a"/>
</p>

<p align="center">
  <br>
    <a href="https://arxiv.org/abs/2404.15121">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href='https://aiganimation.github.io/CAMDM/'>
      <img src='https://img.shields.io/badge/CAMDM-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
    <a href='https://aiganimation.github.io/CAMDM/'>
      <img src='https://img.shields.io/badge/Unity-EXE-57b9d3.svg?style=for-the-badge&logo=unity' alt='Unity'></a>
    <a href="https://youtu.be/J9L0fR_x5OA"><img alt="youtube views" title="Subscribe to my YouTube channel" src="https://img.shields.io/youtube/views/J9L0fR_x5OA?logo=youtube&labelColor=ce4630&style=for-the-badge"/></a>
  </p>

## Update log

- (2024.04.24)
  - Release the Windows Unity demo (GPU) trained in 100STYLE dataset.
- (2024.06.23)
  - Release the training code in PyTorch.
- (2024.07.05)
  - Release the inference code in Unity

## Getting Started

Our project is developed with Unity, and features a real-time character control demo that generates high-quality and diverse character animations, responding in real-time to user-supplied control signals. With our character controller, you can control your character to move with any arbitrary style you want, all achieved through a single unified model.

A well-designed diffusion model is powering behind the demo, and it can be run efficiently on consumer-level GPUs or Apple Silicon MacBooks. For more information, please visit our project's [homepage](https://aiganimation.github.io/CAMDM/) or the [releases page](https://github.com/AIGAnimation/CAMDM/releases) to download the runnable program.

<p align="center">
  <img src="https://github.com/AIGAnimation/CAMDM/assets/7709951/0f2e9940-9920-4e49-8ae3-ce2b6c9c1726"/>
</p>

### Usage

```
WASD:  Move the character and control the character.
F:     Switch between forward mode and orientation-fixed mode. 
QE:    Adjust the orientation in orientation-fixed mode.
J:     Next style
L:     Previous style
Left Shift: Run
```

## Train from Scratch

Our project contains two main modules: Network training part with [PyTorch](https://github.com/AIGAnimation/CAMDM/tree/main/PyTorch), and demo with [Unity](https://github.com/AIGAnimation/CAMDM/tree/main/Unity). Both modules are open-sourced and can be accessed in this repository.

### Character and Motion Preparation

To train a character animation system, you first need a rigged character and its corresponding motion data. In our project, we provide an example with Y-Bot from [Mixamo](https://www.mixamo.com/#/), which uses the standard Mixamo skeleton configuration. We also retargeted the 100STYLE dataset with the Mixamo skeleton. Therefore, you can download any other character from Mixamo and drive it with our trained model.

For customized character and motion data, please wait for our further documentation to explain the retargeting and rigging process.

### Diffusion Network Training [[PyTorch]](https://github.com/AIGAnimation/CAMDM/tree/main/PyTorch) 

All the training codes and documents can be found in the subfolder of our repository.

A practical training session using the entire 100STYLE dataset will take approximately one day, although acceptable checkpoints can usually be obtained after just a few hours (more than 4 hours). Following the completion of the network training, it's necessary to convert the saved checkpoints into the ONNX format. This allows them to be imported into Unity for use as a learning module. For more details, please check the subfolder.

### Unity Inference [[Unity]](https://github.com/AIGAnimation/CAMDM/tree/main/Unity) 
We uses 3060 GPU in the paper

[Youtube tutortial](https://www.youtube.com/watch?v=nuyqpqT3F-A)

## ToDo-List

- [X] Release Unity .exe demo. （2024.04.24）
- [X] Release the training code in PyTorch. （2024.06.23）
- [X] Release the inference code in Unity. （2024.07.05）
- [ ] Release the evaluation code. （TBA）
- [ ] Release the inference code to support any character control. (TBA)

## Acknowledgement

This project is inspired by the following works. We appreciate their contributions, and please consider citing them if you find our project helpful.

- [100STYLE](https://www.ianxmason.com/100style/)
- [AI4Animation](https://github.com/sebastianstarke/AI4Animation) 
- [Guided-Diffusion](https://github.com/openai/guided-diffusion)
- [MDM](https://github.com/GuyTevet/motion-diffusion-model)


## BibTex

```
@inproceedings{camdm,
  title={Taming Diffusion Probabilistic Models for Character Control},
  author={Rui Chen and Mingyi Shi and Shaoli Huang and Ping Tan and Taku Komura and Xuelin Chen},
  booktitle={SIGGRAPH},
  year={2024}
}
```

## Copyright Information
The unity code is released under the GPL-3 license, the rest of the source code is released under the Apache License Version 2.0
