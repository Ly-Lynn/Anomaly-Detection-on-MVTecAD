import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image
import os 
from time import time
import numpy as np

class ModuleClassifier(nn.Module):
    def __init__(self, H=256, W=256, num_classes=15):
        super(ModuleClassifier, self).__init__()

        self.H = H
        self.W = W
        self.classifier = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # (B, 64, H, W)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 64, H/2, W/2)
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), # (B, 32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 32, H/4, W/4)
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), # (B, 16, H/4, W/4)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 16, H/8, W/8)
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1), # (B, 8, H/8, H/8)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (B, 8, H/16, W/16)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * (self.H//16) * (self.W//16), num_classes),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        x = self.fc(x)
        return x

class AnomlayClassifier(nn.Module):
    def __init__(self, H=256, W=256):
        super(AnomlayClassifier, self).__init__()

        self.H = H
        self.W = W
        self.classifier = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # (B, 64, H, W)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 64, H/2, W/2)
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), # (B, 32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 32, H/4, W/4)
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), # (B, 16, H/4, W/4)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 16, H/8, W/8)
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1), # (B, 8, H/8, H/8)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (B, 8, H/16, W/16)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * (self.H//16) * (self.W//16), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        x = self.fc(x)
        return x
        

class ModuleClassifierInference:
    def __init__(self, ckp_path):
        self.model = ModuleClassifier(H=256, W=256)
        self.model.load_state_dict(torch.load(ckp_path, weights_only=True, map_location=torch.device('cpu')))
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.labels = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        self.model.eval()

    def inference(self, img_info):
        img_path = img_info['image_path']
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(img)
            pred = torch.argmax(pred, dim=1)
        return self.labels[pred.item()]
    
class AnomalyClassifierInference:
    def __init__(self, ckp_root, name_module='bottle', outdir='./output'):
        self.model = AnomlayClassifier(H=256, W=256)
        self.ckp_path = os.path.join(ckp_root, 'classification', f'best_anomaly_detection_{name_module}.pth')
        self.model.load_state_dict(torch.load(self.ckp_path, weights_only=True, map_location=torch.device('cpu')))
        self.model.eval()
        self.result = None
        self.outdir = os.path.join(outdir, 'classification')
        os.makedirs(self.outdir, exist_ok=True)
    
    def inference(self, image_info, thres_cls=0.5):
        start = time()

        preprocess = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        GT_path = image_info['groundtruth_path']
        image_path = image_info['image_path']
        image_name = image_info['image_name']
        label = image_info['label']

        image = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            res = self.model(img_tensor)
        end = time()
        if res.item() > thres_cls:
            pred = 1
        else:
            pred = 0
        self.result = {
            "image_name": image_name,
            "image_path": image_path,
            "groundtruth_path": GT_path,
            "label": label,
            "prediction": {
                "pred_labels": pred
            },
            "time": end - start,
            "fig_masks": None,
            "fig_roc_curve": None,
            "auroc": None
        }




