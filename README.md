# MAOSTE: Multimodal Aspect-Object-Sentiment Triples Extraction

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the TMM paper
**MAOSTE: Multimodal Aspect-Object-Sentiment Triples Extraction**.

## 📝 Overview
Multimodal Aspect-Object-Sentiment Triples Extraction (MAOSTE) jointly extracts **aspect terms**, **object entities**, and **sentiment polarities** from image-text pairs. This repository is the official PyTorch implementation of the work, organized for reproducibility and clarity.

## 🛠️ Environment Setup

### 1. Create a virtual environment
```bash
# Create a virtual environment named maoste
conda create -n maoste python=3.8
conda activate maoste
```

### 2. Install dependencies
```bash
# Install core dependencies
pip install -r requirements.txt
```

### 3. Install the package
```bash
python setup.py install
```

## 📦 Dataset Preparation

### 1. Environment Variable
Before running the code, set `PYTHONPATH` to enable module imports:
```bash
cd MAOSTE
export PYTHONPATH=.
```

### 2. Directory Structure
Download the dataset resources and place everything under `original_dataset/`.

Required sources:
- `train.txt`, `dev.txt`, `test.txt`:
  download from
  <https://drive.google.com/drive/folders/1PoZT_95SuNG_l17rp_xYIYO4zppnH8x6?usp=sharing>
  (`maoste/` for the standard split, or `maoste-policy/` for the policy-domain split).
- `image/`:
  download each tweet's associated images from
  <https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view>
- `image_VinVL_Object_36/`:
  download from either
  <https://drive.google.com/drive/folders/1w7W4YYeIE6bK2lAfqRtuwxH-tNqAytiK?usp=sharing>
  or
  <https://pan.baidu.com/s/1QqjOlAAjCqAk_qL6ejeARw?pwd=TwVi>
- `other files`:
  download from
  <https://drive.google.com/drive/folders/1PoZT_95SuNG_l17rp_xYIYO4zppnH8x6?usp=sharing>

A typical layout is:

```
original_dataset/
├── train.txt
├── dev.txt
├── test.txt
├── image/
├── image_VinVL_Object_36/
├── xml/
├── cropped_images/
├── image_anp_output.json
├── image_caption_output.json
└── region_anp_vocab.yaml
```

### 3. Automated Preprocessing Script
```bash
# configure your own src_root / dst_root in generate_dataset.py
python maoste/data/generate_dataset.py
```
After preprocessing, each sample is saved as one `.npz` file.

## 🚀 Quick Start
After dataset preparation, make sure the dataset roots in `maoste/core.py`
point to your preprocessed `.npz` files:

```python
# In maoste/core.py
train_root: str = 'original_dataset/npzfiles/train'
val_root: str = 'original_dataset/npzfiles/dev'
test_root: str = 'original_dataset/npzfiles/test'
```

Then use the scripts below for training or evaluation.

### Training
Use the project training script:
```bash
bash maoste/tasks/mm_gen/train_project.sh
```

### Evaluate on the test split dataset
If you already have trained weights and only want to evaluate on the test set, use:
```bash
bash maoste/tasks/mm_gen/Inference.sh
```

`Inference.sh` uses the same experiment settings as `train_project.sh`, but adds
`--skip-train` so that only evaluation is performed.

Make sure the trained checkpoints are stored under the same `--output-root-dir`
pattern used by the script, such as `e0`, `e1`, and so on. The code will search
for checkpoints under:

```text
<output-root-dir>/MM_GEN/NONE/t5-base/BIO_SENTIMENT/f1/checkpoint-*/pytorch_model.bin
```

For example:

```text
e0/MM_GEN/NONE/t5-base/BIO_SENTIMENT/f1/checkpoint-4000/pytorch_model.bin
```

Important:
- `Inference.sh` does not train a model. It looks for saved checkpoints under the experiment directory and runs evaluation on the test dataset.
- The evaluation dataset path is taken from `test_root` in `maoste/core.py`, so confirm that it points to your preprocessed test `.npz` files before running inference.

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🔍 Citation
This paper was accepted by IEEE Transactions on Multimedia (TMM) in December 2025.
If you use this code in your research, please cite this work. You can add the
BibTeX entry here once the final publication metadata is available.

## 🙏 Acknowledgements
- Using the dataset means you have read and accepted the copyrights set by Twitter and the original dataset providers.
- The dataset construction and parts of this codebase are based on [GMNER](https://github.com/NUSTM/GMNER) and [FMNERG](https://github.com/NUSTM/FMNERG). We sincerely thank the authors for releasing their valuable work.
