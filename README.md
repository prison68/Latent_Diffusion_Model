# Latent Diffusion Model
This repository contains Latent Diffusion Model(LDM) that can be trained from scratch. [stablediffusion](https://github.com/Stability-AI/StableDiffusion?tab=readme-ov-file) repository only provides inference scripts. I have added training scripts and provided two performance metrics for the text2img task.

[LDMCode_Learing_Record.md](./LDMCode_Learing_Record.md) is my learning record of LDM network architecture, for reference only.

## Installation
**Step 1: Clone this repository:**
```bash
git clone https://github.com/prison68/Latent_Diffusion_Model.git
cd Latent_Diffusion_Model
```
**Step 2: Environment Setup:**

Create and activate a new conda environment.

```bash
conda create -n LDM
conda activate LDM
```
**Step 3: Install Dependencies:**

Install the required dependencies with the supported versions.
```
pip install -r requirements.txt
```

## Preparation

**step 1: Download Datasets:**

Please follow the [official website of the LAION dataset](https://laion.ai/blog/laion-400-open-dataset/) to download LAION-400M or use your own dataset.

Assume the datasets is in `./datasets`. It should be like this:
```text
    -00000001.jpg
    -00000001.txt containing the caption
    -00000001.json containing metadata such as the URL, the original width, the EXIF data, whether the image is NSFW
    -00000002.jpg
    -00000002.txt 
    -00000002.json 
```
**step 2: Download Weights:**

Download the weights for [SD2.1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).

## Training
To train on the dataset, you can simply run the script:
```bash
python train.py
```
Note: 
- You can modify the parameters in the `dict` dictionary to adjust the training process.
- All model parameters are defined in `configs/stable-diffusion/v2-inference.yaml`.
- All generated images will be saved to the path `outputs/outputs_imgs`. We will also save a grid of all images.

## Sampling

To generate an image, you can simply run the script:

```bash
python sampler.py
```

## Metrics

This repository also provide the code of the quantitative experiments in `outputs/`.

- `outputs/compute_clip_similarity.py` provides the code needed for computing the image-based CLIP similarities, which computes the CLIP-space similarities between the generated images and the guiding text prompt.
- `outputs/blip_captioning_and_clip_similarity.py` provides the code needed for computing the text-based CLIP similarities which generates captions for each generated image using BLIP and compute the CLIP-space similarities between the generated captions and the guiding text prompt. 

  - Note: to run this script you need to install the library `lavis`. This can be done using `pip install lavis`.

To run the scripts, you simply need to pass the output directory containing the generated images. The direcory structure should be as follows:
```text
outputs/outputs_imgs
|-- prompt_1/
|   |-- 00000.png 
|   |-- 00001.png
|   |-- ...
|   |-- 00064.png
|-- prompt_2/
|   |-- 00000.png 
|   |-- 00001.png
|   |-- ...
|   |-- 00064.png
```
The scripts will iterate through all the prompt outputs provided in the root output directory and aggregate results across all images.

The metrics will be saved to a file under the path specified by `outputs/metrics`.

## Acknowledgements
This code is builds on the code from the [stablediffusion](https://github.com/Stability-AI/StableDiffusion?tab=readme-ov-file) as well as the [Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite) codebase.








