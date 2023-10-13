# DrivingDiffusion
The first multi-view driving scene video generator.

## [Project Page](https://drivingdiffusion.github.io) | [Paper](https://arxiv.org/pdf/2310.07771.pdf)

### [DrivingDiffusion] Training Pipeline
<img width="907" alt="main" src="https://github.com/DrivingDiffusion/DrivingDiffusion.github.io/blob/main/static/images/main.png">

Consistency Module & Local Prompt

<img width="500" alt="main" src="https://github.com/DrivingDiffusion/DrivingDiffusion.github.io/blob/main/static/images/detail.png">

### [DrivingDiffusion] Long Video Generate Pipeline
<img width="907" alt="main" src="https://github.com/DrivingDiffusion/DrivingDiffusion.github.io/blob/main/static/images/inference.png">

### [DrivingDiffusion-Future] Future Generate Pipeline
<img width="780" alt="main" src="https://github.com/DrivingDiffusion/DrivingDiffusion.github.io/blob/main/static/images/future_pipe2.png">


## Abstract 

With the increasing popularity of autonomous driving based on the powerful and unified bird's-eye-view (BEV) representation, a demand for high-quality and large-scale multi-view video data with accurate annotation is urgently required. However, such large-scale multi-view data is hard to obtain due to expensive collection and annotation costs. To alleviate the problem, we propose a spatial-temporal consistent diffusion framework DrivingDiffusion, to generate realistic multi-view videos controlled by 3D layout. There are three challenges when synthesizing multi-view videos given a 3D layout: How to keep 1) cross-view consistency and 2) cross-frame consistency? 3) How to guarantee the quality of the generated instances? Our DrivingDiffusion solves the problem by cascading the multi-view single-frame image generation step, the single-view video generation step shared by multiple cameras, and post-processing that can handle long video generation. In the multi-view model, the consistency of multi-view images is ensured by information exchange between adjacent cameras. In the temporal model, we mainly query the information that needs attention in subsequent frame generation from the multi-view images of the first frame.  We also introduce the local prompt to effectively improve the quality of generated instances. In post-processing, we further enhance the cross-view consistency of subsequent frames and extend the video length by employing temporal sliding window algorithm. Without any extra cost, our model can generate large-scale realistic multi-camera driving videos in complex urban scenes, fueling the downstream driving tasks. The code will be made publicly available.
<img width="907" alt="abs" src="https://github.com/DrivingDiffusion/DrivingDiffusion.github.io/blob/main/static/images/intro.png">


## News&Logs
- **[2023/8/15]** Single-View future generation.
- **[2023/5/18]** Multi-View video generation controlled by 3D Layout.
- **[2023/3/01]** Multi-View image generation controlled by 3D Layout.
- **[2023/3/01]** Single-View image generation controlled by 3D Layout.
- **[2023/2/03]** Single-View image generation controlled by Laneline Layout.


## Usage

### Setup Environment
``` 
conda create -n dridiff python=3.8
conda activate dridiff
pip install -r requirements.txt
``` 
**DrivingDiffusion** is training on 8 A100. 

### Weight
We use the **stable-diffsuion-v1-4** initial weights and base structure. Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. For more information about how Stable Diffusion functions, please have a look at 🤗's Stable Diffusion with 🧨Diffusers blog, which you can find at [**HuggingFace**](https://huggingface.co/CompVis/stable-diffusion-v1-4)

### Data Preparation
#### nuScenes
#### Custom Dataset


### Training
Coming soon...
### Inference
Coming soon...


## Results
### Visualization of Multi-View Image Generation.
<div align="center">   
<img width="750" alt="abs" src="https://github.com/DrivingDiffusion/DrivingDiffusion.github.io/blob/main/static/images/multiview_img.png">
</div>

### Visualization of Temporal Generation.
<div align="center">   
<img width="750" alt="abs" src="https://github.com/DrivingDiffusion/DrivingDiffusion.github.io/blob/main/static/images/temporal_img_f.png">
</div>

### Visualization of Control Capability.
<div align="center">   
<img width="907" alt="abs" src="https://github.com/DrivingDiffusion/DrivingDiffusion.github.io/blob/main/static/images/control.png">
</div>

### Multi-View Video Generation of Driving Scenes Controlled by 3D Layout

[**Videos**](https://drivingdiffusion.github.io) 

### Ability to Construct future
#### Control future video generation through text description of road conditions
<div align="center">   
<img width="907" alt="abs" src="https://github.com/DrivingDiffusion/DrivingDiffusion.github.io/blob/main/static/images/future.png">
</div>

#### Future video generation without text description of road conditions
<div align="center">   
<img width="907" alt="abs" src="https://github.com/DrivingDiffusion/DrivingDiffusion.github.io/blob/main/static/images/future_unc.png">
</div>

## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.

