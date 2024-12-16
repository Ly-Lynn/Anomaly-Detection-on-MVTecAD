import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from config import  OUTDIR

def load_mask_from_file(mask_path, target_shape=(224,224)):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ground_truth_mask = (mask > 0).astype(np.uint8)
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.numpy()
    
    resized_mask = cv2.resize(
        ground_truth_mask, 
        (target_shape[-1], target_shape[-2]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    return resized_mask

def calculate_mask_auroc(  ground_truth_path, pred_masks):
    if not ground_truth_path:
        return
    ground_truth_mask = load_mask_from_file(ground_truth_path, target_shape=(pred_masks.shape[-2], pred_masks.shape[-1]))
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.numpy()
    
    pred_mask_np = pred_masks.numpy().squeeze()
    
    ground_truth_mask = (ground_truth_mask > 0).astype(int)
    pred_mask_np = pred_mask_np.astype(int)
    
    gt_flat = ground_truth_mask.flatten()
    pred_flat = pred_mask_np.flatten()
    print("SHAPE", ground_truth_mask.shape, pred_mask_np.shape)
    auroc = roc_auc_score(gt_flat, pred_flat)
    
    return auroc


def visualize_mask_roc_curve(result, ground_truth_path, pred_masks):
    if not ground_truth_path:
        return
    ground_truth_mask = load_mask_from_file(ground_truth_path, target_shape=(pred_masks.shape[-2], pred_masks.shape[-1]))
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.numpy()
    
    pred_mask_np = pred_masks.numpy().squeeze()
    
    ground_truth_mask = (ground_truth_mask > 0).astype(int)
    pred_mask_np = pred_mask_np.astype(int)
    
    gt_flat = ground_truth_mask.flatten()
    pred_flat = pred_mask_np.flatten()
    
    fpr, tpr, thresholds = roc_curve(gt_flat, pred_flat)
    auroc = roc_auc_score(gt_flat, pred_flat)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve\nfor Anomaly Mask Detection')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.savefig(OUTDIR + '/roc_curve.png')
    result['fig_roc_curve'] = OUTDIR + '/roc_curve.png'
    result['auroc'] = auroc
    plt.close()
    return result

def visualize_masks(result, ground_truth_path, pred_masks):
    if not ground_truth_path:
        return
    ground_truth_mask =  load_mask_from_file(ground_truth_path, target_shape=(pred_masks.shape[-2], pred_masks.shape[-1]))
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.numpy()
    
    pred_mask_np = pred_masks.numpy().squeeze()
    
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8) * 255
    pred_mask_np = pred_mask_np.astype(np.uint8) * 255
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth Mask')
    plt.imshow(ground_truth_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(pred_mask_np, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    
    plt.savefig( OUTDIR + '/masks.png')
    result['fig_masks'] =  OUTDIR + '/masks.png'
    plt.close()
    return result