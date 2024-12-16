from pathlib import Path
import os
from pathlib import Path
from anomalib import TaskType
from anomalib.data import MVTec, PredictDataset
from PIL import *
from tqdm import tqdm
import base64

dataset_root = Path.cwd() / "datasets" / "MVTec"
module = MVTec(
    root=dataset_root,  
            category="bottle",
            image_size=256,     
            train_batch_size=16,  
            eval_batch_size=16,   
            num_workers=4,  
)
module.prepare_data()
print(dataset_root)