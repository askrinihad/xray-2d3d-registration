"""
ARCADE dataset loader: parses the COCO-style JSON annotations and
rasterizes polygon segmentations into binary vessel masks.

For a first working version, all 26 SYNTAX vessel-segment categories are
collapsed into a single foreground class (binary vessel-vs-background
segmentation). Multi-class segmentation (26 SYNTAX regions) is a natural
extension once this baseline works -- see the README roadmap.
"""
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ArcadeSegmentationDataset(Dataset):
    def __init__(self, root, split="train", task="syntax", image_size=256):
        self.image_dir = os.path.join(root, task, split, "images")
        ann_path = os.path.join(root, task, split, "annotations", f"{split}.json")
        with open(ann_path) as f:
            coco = json.load(f)

        self.image_size = image_size
        self.images = {img["id"]: img for img in coco["images"]}
        self.anns_by_image = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        info = self.images[image_id]

        img_path = os.path.join(self.image_dir, info["file_name"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        # Rasterize every polygon (across all 26 vessel categories) into one
        # binary foreground mask.
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in self.anns_by_image.get(image_id, []):
            for seg in ann.get("segmentation", []):
                pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], color=1)

        img = cv2.resize(img, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        return img_t, mask_t
