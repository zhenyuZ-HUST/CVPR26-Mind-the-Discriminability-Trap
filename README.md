# **Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning**

Official implementation for **CVPR 2026** poster paper: *"**Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning**"*.

## 📋 Overview

This repository discovers the visual discriminative trap in VLM fine-tuning and proposes a solution to suppress this trap, promoting cross-modal learning during the VLM fine-tuning.

## 🛠️ Environment Setup

**Note**: Creating a new environment requires matching CUDA versions. The code was developed with CUDA 11.8.

#### 1. Create Conda Environment

```bash
conda create -n Mind python=3.10 -y
conda activate Mind
```

#### 2. Install PyTorch with CUDA Support

```bash
# For CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Install CLIP (OpenAI)

```bash
pip install git+https://github.com/openai/CLIP.git
```

## 📦 Datasets

- **EuroSAT**: Satellite imagery land cover classification
- **CropDisease**: Plant disease recognition
- **ISIC**: Skin lesion classification (dermatology)
- **ChestX**: Chest X-ray disease classification

Please prepare your datasets in the appropriate directory structure before running experiments.

## 🚀 Usage

### Training Commands

**For EuroSAT and CropDisease:**

```bash
# 1-shot setting
python main.py --encoder vision --r 16 --alpha 8 --epochs 250 --shot 1 --episodes 800 --dataset EuroSAT/CropDisease/ISIC/ChestX

# 5-shot setting
python main.py --encoder vision --r 16 --alpha 8 --epochs 250 --shot 5 --episodes 400 --dataset EuroSAT/CropDisease/ISIC/ChestX

```



### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (EuroSAT/CropDisease/ISIC/ChestX) | ISIC |
| `--shot` | Number of support examples per class (1 or 5) | 5 |
| `--way` | Number of classes per episode | 5 |
| `--episodes` | Number of test episodes | 400 |
| `--epochs` | Training epochs per episode | 250 |
| `--encoder` | Which encoder to fine-tune (vision/text/both) | vision |
| `--r` | LoRA rank | 16 |
| `--alpha` | LoRA scaling factor | 8 |
| `--lr` | Learning rate for LoRA parameters | 2e-4 |
| `--beta` | weight for anti-visual loss | 0.1 |
| `--gamma` | weight for relationship alignment | 3 |

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{zhang2026mind,
  title={Mind the discriminability trap in source-free cross-domain few-shot learning},
  author={Zhang, Zhenyu and Zou, Yixiong and Li, Yuhua and Li, Ruixuan and Chen, Guangyao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={36978--36988},
  year={2026}
}
```

## 🙏 Acknowledgments

This repository is developed based on:
- [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA)
- [StepSPT](https://github.com/xuhuali-mxj/StepSPT)

We thank the authors for their excellent codebases.
