
````markdown
# Easy Annotation

This repository provides annotation utilities for EndoViT-based models, along with dataset handling and pre-trained weight loading.

## 🛠️ Environment Setup

Clone the repository and move into the project directory:

```bash
git clone https://github.com/yonding/easy_annotation.git
cd easy_annotation
````

### Option 1: Install with `requirements.txt`

```bash
pip install -r requirements.txt
```

### Option 2: Create Conda environment with `environment.yml`

```bash
conda env create -f environment.yml
conda activate easy_annotation
```

## 📦 Prepare EndoViT Weights

Copy the EndoViT model checkpoint:

```bash
cp /data/kayoung/repos/graph_wo_detector/easy_annotation/endovit/pytorch_model.bin ./endovit
```

## 📂 Prepare Dataset

Copy the Endoscapes-SG201 dataset:

```bash
cp -rv /data/kayoung/repos/graph_wo_detector/easy_annotation/Endoscapes-SG201 ./
```

