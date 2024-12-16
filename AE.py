import torch
import torch.nn as nn
import math
import os
from time import time
from PIL import Image
import torchvision.transforms as transforms
from visualization import visualize_masks, calculate_mask_auroc, visualize_mask_roc_curve

class Autoencoder(nn.Module):
    def __init__(self, input_channels=3, H=256, W=256):
        super(Autoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.H = H
        self.W = W
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AEInference:
    def __init__(self, ckp_root, name_module='bottle', outdir='./output'):
        self.model = Autoencoder(H=256, W=256)
        self.ckp_path = os.path.join(ckp_root, 'ae', f'best_aes_{name_module}.pth')
        self.model.load_state_dict(torch.load(self.ckp_path, weights_only=True, map_location=torch.device('cpu')))
        self.model.eval()
        self.result = None

    def inference(self, image_info, thres_reconstruction = 0.005, thres_mask=0.01):
        GT_path = image_info['groundtruth_path']
        image_path = image_info['image_path']
        image_name = image_info['image_name']
        label = image_info['label']
        start = time()
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            reconstructed_image = self.model(image)
        end = time()
        reconstruction_error = (image - reconstructed_image) ** 2
        anomaly_mask = torch.mean(reconstruction_error, dim=1, keepdim=True) > thres_mask
        predictions = {
            'anomaly_maps': reconstruction_error,
            'pred_masks': anomaly_mask,
            'pred_labels': torch.mean(reconstruction_error) > thres_reconstruction
        }
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
