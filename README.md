# Towards Fine-Grained Text-to-3D Quality Assessment: A Benchmark and A Two-Stage Rank-Learning Metric
<div align="center"> Bingyang Cui<sup>1*</sup>, Yujie Zhang<sup>1*</sup>, Qi Yang<sup>2</sup>, Zhu Li<sup>2</sup>, Yiling Xu<sup>1?</sup>

<div align="center"> <sup>1</sup> Shanghai Jiao Tong University, <sup>2</sup> University of Missouri-Kansas City</small>
<div align="center"> <small><sup>*</sup> Equal Contribution &nbsp;&nbsp;&nbsp <sup>?</sup>Corresponding author</small>

<div align="center">
  <a href="https://link.springer.com/article/10.1007/s11263-026-02994-x?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20260825&utm_content=10.1007/s11263-026-02994-x" target="_blank"><img src="https://img.shields.io/badge/Paper_Link-IJCV-blue"></a>
  <a href="https://arxiv.org/abs/2509.23841" target="_blank"><img src="https://img.shields.io/badge/Paper_PDF-arXiv-red"></a>
  <a href="https://cbysjtu.github.io/Rank2Score/" target='_blank'><img src="https://img.shields.io/badge/Project-&#x1F680-blue"></a>
</div>

This repository contains a PyTorch implementation of **Rank2Score**, as presented in our paper [*Towards Fine-Grained Text-to-3D Quality Assessment: A Benchmark and A Two-Stage Rank-Learning Metric*](https://link.springer.com/article/10.1007/s11263-026-02994-x?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20260825&utm_content=10.1007/s11263-026-02994-x).


## 🔥 News

- **[2026.08]** Our paper has been accepted by **IJCV** 🎉🎉🎉


## 📖 Overview

**T23D-CompBench** is a comprehensive benchmark for compositional Text-to-3D generation. **Rank2Score** is an effective evaluator with two-stage training for Text-to-3D quality assessment.

> Recent advances in Text-to-3D generative models have enabled the synthesis of diverse, high-fidelity 3D assets from textual prompts. However, existing challenges restrict the development of reliable T23D quality assessment. To address the existing limitations, we introduce T23D-CompBench, a comprehensive benchmark for compositional T23D generation. We define five components with twelve sub-components for compositional prompts, which are used to generate 3,600 textured meshes from ten state-of-the-art generative models. A large-scale subjective experiment is conducted to collect 129,600 reliable human ratings across different perspectives. Based on T23D-CompBench, we further propose Rank2Score, an effective evaluator with two-stage training for Text-to-3D quality assessment. Rank2Score enhances pairwise training via supervised contrastive regression and curriculum learning in the first stage, and subsequently refines predictions using mean opinion scores to achieve closer alignment with human judgments in the second stage. Extensive experiments and downstream applications demonstrate that Rank2Score consistently outperforms existing metrics across multiple dimensions and can additionally serve as a reward function to optimize generative models.

<div align="center">
<img src="https://github.com/cbysjtu/rank2score_code/blob/main/asset/framework.png" width = 80% height = 80%/>
<br>
Overview of the Rank2Score Evaluator
</div>


## 🛠️ Installation

To set up this repository, clone it, create a new conda environment, and install all dependencies within it:

```bash
# Clone this repository
git clone https://github.com/cbysjtu/rank2score_code.git
cd rank2score_code

# Create and activate a new conda environment (Python 3.10+)
conda create --name Rank2Score python=3.10 -y
conda activate Rank2Score 

# Install dependencies
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

# Additionally, we render texture meshed into images by Pytorch3D, please follow the steps to install Pytorch3D.
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

## 📑 Open-Source Plan

- [ ] Code Usage
- [ ] Model Checkpoints


## 📝 Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{cui2026rank2score,
author = {Cui, Bingyang and Zhang, Yujie and Yang, Qi and Li, Zhu and Xu, Yiling},
journal = {International Journal of Computer Vision (IJCV)},
title = {Towards Fine-Grained Text-to-3D Quality Assessment: A Benchmark and A Two-Stage Rank-Learning Metric},
year = {2026}
}
```
