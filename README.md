# CLIP-DisDiff: CLIP-Guided Disentanglement of Diffusion Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository extends **DisDiff** [[paper]](https://arxiv.org/pdf/2301.13721) [[code]](https://github.com/ThomasMrY/DisDiff)—a method for disentangling diffusion probabilistic models—by integrating **OpenAI's CLIP** for multimodal guidance. **CLIP-DisDiff** leverages both text and image encoders from CLIP to achieve more flexible and controllable latent factor disentanglement in diffusion models.

---

## Overview

### What is DisDiff?
**DisDiff** disentangles the gradient fields of a pretrained diffusion probabilistic model (e.g., DDPM) into separate sub-gradients, each corresponding to a latent factor (or concept). Once disentangled, you can individually manipulate these factors for more targeted image generation.

### How Does CLIP-DisDiff Differ?
**CLIP-DisDiff** replaces the original image-based encoder with **CLIP's text encoder** (and optionally the image encoder) to guide the disentanglement process. Specifically:

1. **Text Attributes**  
   Instead of splitting a single image into multiple latent vectors, we directly feed text attributes (e.g., color, style) into the CLIP text encoder. Each attribute is encoded into a latent vector, naturally mapping "one text attribute → one latent vector."

2. **Decoder with Conditioned Gradients**  
   Each latent vector from CLIP is passed into the decoder (G_ψ), which computes the conditioned gradients for diffusion steps. This results in separate image predictions reflecting each text attribute.

3. **Disentangling Loss**  
   Similar to DisDiff, a disentangling loss encourages independence among latent factors. However, here we can also leverage CLIP's image encoder to align text embeddings and generated images more explicitly (optional).

<p align="center">
  <img src="./assets/clip_disdiff_arch.png" width="60%"><br>
  <em>Figure 1: High-level architecture of CLIP-DisDiff.</em>
</p>

---

## Requirements

A sample Conda environment file is included (`environment.yaml`). To create and activate the environment:

```bash
conda env create -f environment.yaml
conda activate clip-disdiff
```

Additionally, install the following:
```bash
pip install torch torchvision
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yu12ki04/CLIP-DisDiff.git
cd CLIP-DisDiff
```

2. Install in editable mode:
```bash
pip install -e .
```

## Usage

### 1. Train a Base Autoencoder (Optional)
If you plan to train a latent diffusion model (LDM) from scratch, you may need a base VAE or VQ-VAE. For instance:

```bash
python main.py --base configs/autoencoder/example_vq_4_16.yaml -t --gpus 0,
```

Adjust the config file as needed.

### 2. Train a Latent Diffusion Model (LDM)
After obtaining a pretrained autoencoder, specify its checkpoint path in the LDM config and run:

```bash
python main.py --base configs/latent-diffusion/example_vq_4_16.yaml -t --gpus 0,
```

### 3. Train CLIP-DisDiff
Specify both the pretrained LDM/VAE checkpoints and the CLIP encoder parameters in your config (e.g., configs/latent-diffusion/example_vq_4_16_clip-dis.yaml). Then:

```bash
python main.py \
  --base configs/latent-diffusion/example_vq_4_16_clip-dis.yaml \
  -t --gpus 0, \
  -l exp_clip_disdiff \
  -n experiment_name \
  -s 0
```

- `-n` sets the experiment name
- `-s` sets a random seed (optional)

### 4. Evaluation
Similar to DisDiff, evaluation can be performed with:

```bash
python run_para_metrics.py -l exp_clip_disdiff -p 10
```

Adjust arguments as needed (e.g., log directory, number of processes).

## Model Configuration Example

Below is a simplified example of a config snippet integrating CLIP:

```yaml
model:
  base_learning_rate: 1e-4
  params:
    # Checkpoint paths
    ckpt_path: "/path/to/pretrained_ldm"
    first_stage_config:
      ckpt_path: "/path/to/pretrained_vae"
    
    # CLIP text encoder
    cond_stage_config:
      text_encoder:
        target: ldm.modules.encoders.modules.FrozenCLIPTextEmbedder
        params:
          version: "ViT-L/14"
    
    # CLIP image encoder
      image_encoder:
        target: ldm.modules.encoders.modules.FrozenClipImageEmbedder
        params:
          model: "ViT-L/14"

    # Disentangling-specific settings
    ...
```

## Model Configuration

The model can be configured through the YAML config files. Key parameters for CLIP integration:

```yaml
model:
  params:
    cond_stage_config:
      text_encoder:
        target: ldm.modules.encoders.modules.FrozenCLIPTextEmbedder
        params:
          version: 'ViT-L/14'
      image_encoder:
        target: ldm.modules.encoders.modules.FrozenClipImageEmbedder
        params:
          model: 'ViT-L/14'
```

## Citation

If you use this code for your research, please cite our work:

```bibtex
@inproceedings{yang2023disdiff,
  title={DisDiff: Unsupervised Disentanglement of Diffusion Probabilistic Models},
  author={Yang, Tao and Wang, Yuwang and Lu, Yan and Zheng, Nanning},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@misc{radford2021clip,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  howpublished={arXiv preprint arXiv:2103.00020},
  year={2021}
}

@article{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  journal={CVPR},
  year={2022}
}
```

## Acknowledgements

This project builds upon:
- [DisDiff](https://github.com/ThomasMrY/DisDiff)
- [CLIP](https://github.com/openai/CLIP)
- [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please:
- Open an issue in this repository
- Contact: [your-email@example.com](mailto:your-email@example.com)

## Repository

The code is available at: [CLIP-DisDiff](https://github.com/yu12ki04/CLIP-DisDiff)