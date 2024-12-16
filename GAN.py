import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
import numpy as np
import math
from time import time
from visualization import visualize_masks, calculate_mask_auroc, visualize_mask_roc_curve

class Generator(nn.Module):
    def __init__(self, noise_dim=114, output_channels=3, H=128, W=128, feature_maps=256):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.output_channels = output_channels
        self.H = H  # Height of the output image
        self.W = W  # Width of the output image
        self.feature_maps = feature_maps
        
        start_size = 4 
        print(H,W)
        self.upsample_steps = int(math.log2(max(H, W)) - math.log2(start_size)) -1
        
        self.noise_projection = nn.Sequential(
            nn.Linear(noise_dim, feature_maps * start_size * start_size),
            nn.ReLU(True)
        )
        
        layers = []
        in_channels = feature_maps
        print(self.upsample_steps)
        for i in range(self.upsample_steps):
            out_channels = max(feature_maps // (2 ** (i + 1)), output_channels)
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            ])
            in_channels = out_channels
        
        layers.append(nn.ConvTranspose2d(in_channels, output_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())
        
        self.generator = nn.Sequential(*layers)

    def forward(self, noise):
        x = self.noise_projection(noise)
        
        x = x.view(-1, self.feature_maps, 4, 4)
        
        x = self.generator(x)
        
        assert x.shape[2] == self.H and x.shape[3] == self.W, \
            f"Output size {x.shape[2:]} does not match expected size ({self.H}, {self.W})"
        
        return x

class Discriminator(nn.Module):
    def __init__(self, H=128, W=128, input_channels = 1):
        super(Discriminator, self).__init__()
        self.input_channels = input_channels
        self.discriminator = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=4, stride=2, padding=1),  # (B, 1, H, W) -> (B, 64, H/2, W/2)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (B, 128, H/4, W/4)
             nn.InstanceNorm2d(128,  affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> (B, 256, H/8, W/8)
            nn.InstanceNorm2d(256,  affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # -> (B, 512, H/16, W/16)
            nn.InstanceNorm2d(512,  affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),  
            nn.Linear(512 * (H // 16) * (W // 16), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.discriminator(x)
        # x = x.view(-1, 1, 2, 2)
        return x
    
class GanInference:
    def __init__(self, ckp_root, name_module='bottle', outdir='./output'):
        self.gen_ckpt = os.path.join(ckp_root, 'gan', f'best_G_{name_module}.pth')
        self.dis_ckp = os.path.join(ckp_root, 'gan', f'best_D_{name_module}.pth')
        self.generator = Generator(noise_dim=256, H=256, W=256)
        self.discriminator = Discriminator(H=256, W=256, input_channels=3)
        self.generator.load_state_dict(torch.load(self.gen_ckpt, weights_only=True, map_location=torch.device('cpu')))
        self.discriminator.load_state_dict(torch.load(self.dis_ckp, weights_only=True, map_location=torch.device('cpu')))
        self.generator.eval()
        self.discriminator.eval()
        self.outdir = os.path.join(outdir, 'gan')
        self.result = None
        os.makedirs(self.outdir, exist_ok=True)
    def inference(self, image_info, thres_cls=0.4, thres_mask=0.005):
        start = time()
        noise = np.random.normal(0, 1, size=[1, 256])
        noise = torch.from_numpy(noise).float()

        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        GT_path = image_info['groundtruth_path']
        image_path = image_info['image_path']
        image_name = image_info['image_name']
        label = image_info['label']

        image = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(image).unsqueeze(0)

        res_G = self.generator(noise)
        res_D = self.discriminator(img_tensor)
        is_anomaly = res_D < thres_cls

        reconstruction_error = (img_tensor - res_G) ** 2  
        anomaly_mask = torch.mean(reconstruction_error, dim=1, keepdim=True) > thres_mask 
        end = time()
        predictions = {
            'anomaly_maps': reconstruction_error,
            'pred_masks': anomaly_mask,
            'pred_labels': is_anomaly
        }
        self.result = {
            "image_name": image_name,
            "image_path": image_path,
            "groundtruth_path": GT_path,
            "label": label,
            "prediction": predictions,
            "time": end - start
        }
    def visualization (self, ground_truth_path, pred_masks):
        self.result = visualize_masks(self.result, ground_truth_path, pred_masks)
        calculate_mask_auroc(ground_truth_path, pred_masks)
        self.result = visualize_mask_roc_curve(self.result, ground_truth_path, pred_masks)