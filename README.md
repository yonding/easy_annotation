# Easy Annotation

## 🛠️ Environment Setup

Clone the repository and move into the project directory:

```bash
git clone https://github.com/yonding/easy_annotation.git
cd easy_annotation
```

Create Conda environment with `environment.yml`

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

## 🌿 Create Your Own Branch

Before running experiments, create and switch to your own branch:

```bash
git branch your_branch_name
git checkout your_branch_name
```

This will keep your experiments isolated from the main branch.
