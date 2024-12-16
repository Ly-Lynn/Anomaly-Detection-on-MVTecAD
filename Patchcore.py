import anomalib
import os
import torch
from pathlib import Path
from anomalib import TaskType
from anomalib.data import MVTec, PredictDataset
from PIL import Image
from tqdm import tqdm
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.utils.post_processing import superimpose_anomaly_map
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from time import time
from sklearn.metrics import roc_auc_score, roc_curve
import cv2
from visualization import load_mask_from_file, calculate_mask_auroc, visualize_mask_roc_curve, visualize_masks

class PatchCore:
    def __init__(self, ckp_root, name_module='bottle', outdir='./output'):
        self.ckpt_path = os.path.join(ckp_root, 'patchcore', f'{name_module}_patchcore.ckpt')

        self.model = Patchcore.load_from_checkpoint(
            self.ckpt_path,
            backbone="wide_resnet50_2",
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=9
        )
        self.callback = [
            ModelCheckpoint(
                mode="max",
                monitor="image_AUROC",
            ),
            EarlyStopping(
                monitor="image_AUROC",
                mode="max",
                patience=3,
            ),
        ]
        self.engine = Engine(
            callbacks=self.callback,
            pixel_metrics="AUROC",
            accelerator="auto",
            devices='auto',
            logger=True,
        )
        self.result = None
        self.outdir = os.path.join(outdir, 'patchcore')
        os.makedirs(self.outdir, exist_ok=True)

    def inference(self, image_info):
        GT_path = image_info['groundtruth_path']
        image_path = image_info['image_path']
        image_name = image_info['image_name']
        label = image_info['label']
        predicted_object = PredictDataset(path=image_path)
        image_loader = DataLoader(dataset=predicted_object)
        start = time()
        predictions = self.engine.predict(model=self.model, dataloaders=image_loader, ckpt_path=self.ckpt_path)[0]
        end = time()
        self.result = {
            "image_name": image_name,
            "image_path": image_path,
            "groundtruth_path": GT_path,
            "label": label,
            "prediction": predictions,
            "time": end - start
        }
    def visualization(self, ground_truth_path, pred_masks):
        self.result = visualize_masks(self.result, ground_truth_path, pred_masks)
        calculate_mask_auroc(ground_truth_path, pred_masks)
        self.result = visualize_mask_roc_curve(self.result, ground_truth_path, pred_masks)
    





