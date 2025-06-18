import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torch import nn
from collections import defaultdict
from finch import FINCH
import types
import random
from sklearn.cluster import KMeans, AgglomerativeClustering
import argparse

# ==================== Argument 추가 ====================
parser = argparse.ArgumentParser()
parser.add_argument('--cluster_method', type=str, choices=['finch', 'kmeans', 'hac'], default='finch',
                    help='클러스터링 방법 선택: finch, kmeans, hac')
parser.add_argument('--kmeans_k', type=int, default=8, help='kmeans 사용 시 클러스터 개수')
parser.add_argument('--hac_k', type=int, default=8, help='HAC 사용 시 클러스터 개수')
parser.add_argument('--linkage', type=str, default='ward', choices=['ward', 'average', 'complete', 'single'],
                    help='HAC linkage 방식')
args = parser.parse_args()

# Paths
data_dir = Path('/data/kayoung/repos/graph_wo_detector/clustering/Endoscapes-SG201/train')
pairs_dir = Path(f'./results/{args.cluster_method}{("_"+args.linkage) if args.cluster_method == "hac" else ""}')
pairs_dir.mkdir(parents=True, exist_ok=True)

# Hyper-parameters
cluster_level = 1
input_size = 224
patch_grid = 14
patch_size = input_size // patch_grid
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

# Data normalization
dataset_mean = [0.3464, 0.2280, 0.2228]
dataset_std = [0.2520, 0.2128, 0.2093]

# Helper functions
def process_single_image(image_path):
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean, std=dataset_std),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img)

def load_vit(device, dtype):
    ckpt = torch.load("./endovit/pytorch_model.bin", map_location=device)['model']
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4., qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    model.load_state_dict(ckpt, strict=False)

    def forward_features_all(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    model.forward_features_all = types.MethodType(forward_features_all, model)
    return model.to(device, dtype).eval()

def generate_colormap(n):
    np.random.seed(0)
    return np.random.randint(0, 256, (n, 3), dtype=np.uint8)

# load model
model = load_vit(device, dtype)

# load images
img_paths = list(data_dir.rglob("*.jpg"))
random.shuffle(img_paths)

for p in img_paths:
    img = Image.open(p).convert('RGB')
    img_resized = img.resize((input_size, input_size))

    tensor = process_single_image(p).unsqueeze(0).to(device, dtype)
    with torch.no_grad():
        feats = model.forward_features_all(tensor).squeeze(0).cpu()[1:]  # (196,768)

    all_feats, metadata = [], []
    for idx in range(patch_grid**2):
        row, col = divmod(idx, patch_grid)
        x0, y0 = col * patch_size, row * patch_size
        patch = img_resized.crop((x0, y0, x0 + patch_size, y0 + patch_size))
        arr = np.array(patch)
        dark = np.all(arr < 20, axis=-1).sum()
        if dark / (patch_size * patch_size) <= 0.1:
            metadata.append({'row': row, 'col': col})
            all_feats.append(feats[idx])

    if not all_feats:
        print(f"⚠️ No valid patches for {p.name}")
        continue

    feats_tensor = torch.stack(all_feats)
    norm_feats = feats_tensor / feats_tensor.norm(dim=-1, keepdim=True)

    # ==================== 클러스터링 선택 ====================
    if args.cluster_method == 'finch':
        clusters, _, _ = FINCH(norm_feats.numpy())
        labels = clusters[:, cluster_level].tolist()
    elif args.cluster_method == 'kmeans':
        kmeans = KMeans(n_clusters=args.kmeans_k, n_init=10, random_state=0).fit(norm_feats.numpy())
        labels = kmeans.labels_.tolist()
    elif args.cluster_method == 'hac':
        hac = AgglomerativeClustering(n_clusters=args.hac_k, linkage=args.linkage)
        labels = hac.fit_predict(norm_feats.numpy()).tolist()
    else:
        raise ValueError(f"Unknown cluster method: {args.cluster_method}")

    unique_labels = np.unique(labels)
    cmap = generate_colormap(len(unique_labels))
    color_map = {u: cmap[i] for i, u in enumerate(unique_labels)}

    clusters_loc = defaultdict(list)
    for meta, lbl in zip(metadata, labels):
        clusters_loc[lbl].append((meta['row'], meta['col']))

    # ✅ Overlay 이미지 생성
    base_overlay = img_resized.convert("RGBA")
    overlay = Image.new("RGBA", base_overlay.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for meta, lbl in zip(metadata, labels):
        r, c = meta['row'], meta['col']
        x0, y0 = c * patch_size, r * patch_size
        x1, y1 = x0 + patch_size, y0 + patch_size
        col = tuple(color_map[lbl].tolist()) + (80,)
        d.rectangle([x0, y0, x1, y1], fill=col)
    overlay_img = Image.alpha_composite(base_overlay, overlay).convert("RGB")

    # ✅ BBox 이미지 생성
    base_rgb = img_resized.copy()
    draw_bb = ImageDraw.Draw(base_rgb)
    for lbl, locs in clusters_loc.items():
        rows, cols = zip(*locs)
        min_x, min_y = min(cols) * patch_size, min(rows) * patch_size
        max_x, max_y = (max(cols) + 1) * patch_size, (max(rows) + 1) * patch_size
        col = tuple(color_map[lbl].tolist())
        draw_bb.rectangle([min_x, min_y, max_x, max_y], outline=col, width=3)

    # ✅ 두 이미지를 나란히 붙여서 저장
    concat_img = Image.new("RGB", (input_size*2, input_size), (255,255,255))
    concat_img.paste(overlay_img, (0,0))
    concat_img.paste(base_rgb, (input_size, 0))

    pair_path = pairs_dir / f"{p.stem}.png"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    concat_img.save(pair_path)

    print(f"✅ Done for {p.name}")
